"""The local character model and the workspace's original SentencePiece demo."""
import argparse
import csv
import io
import math
import random
import time
import unicodedata
import zipfile
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import sentencepiece as spm
import torch

from .model import ModelConfig, TinyGPT

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/1000-data-science-concepts.zip"


class Characters:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.indices = {ch: i for i, ch in enumerate(vocabulary)}

    def encode(self, text):
        return [self.indices.get(ch, 0) for ch in text]

    def decode(self, ids):
        return "".join(self.vocabulary[i] for i in ids if i > 1)


def corpus():
    def normalize(value):
        return " ".join(unicodedata.normalize("NFKC", value or "").split())

    with zipfile.ZipFile(ARCHIVE) as archive:
        with archive.open("data_science_concepts.csv") as member:
            rows = csv.DictReader(io.TextIOWrapper(member, encoding="utf-8-sig"))
            docs = {}
            for row in rows:
                q, a = normalize(row["Question"]), normalize(row["Answer"])
                if q and a:
                    docs.setdefault(q.casefold(), f"Question: {q}\nAnswer: {a}\n")
    docs = list(docs.values())
    random.Random(42).shuffle(docs)
    first, second = len(docs) * 8 // 10, len(docs) * 9 // 10
    splits = {"train": docs[:first], "validation": docs[first:second], "test": docs[second:]}
    tokenizer = Characters(["<unk>", "<end>"] + sorted(set("".join(splits["train"]))))
    tokens = {name: torch.tensor([i for doc in group for i in [*tokenizer.encode(doc), 1]])
              for name, group in splits.items()}
    return tokenizer, tokens, {name: len(group) for name, group in splits.items()}


def batch(tokens, length, count, rng):
    starts = torch.randint(len(tokens) - length, (count, 1), generator=rng)
    positions = starts + torch.arange(length)
    return tokens[positions], tokens[positions + 1]


@torch.inference_mode()
def aggregate(model, tokens):
    """Fixed windows allow comparisons at the same positions across checkpoints."""
    rng = torch.Generator().manual_seed(177)
    score = 0.0
    for _ in range(4):
        x, y = batch(tokens, model.config.block_size, 8, rng)
        _, loss = model(x, y)
        score += loss.item()
    return score / 4


def write_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        tmp = Path(temp.name)
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def train(path, steps=500, seed=42, progress=lambda event: None, stopped=lambda: False):
    tokenizer, tokens, counts = corpus()
    cfg = ModelConfig(len(tokenizer.vocabulary), 128, 64, 4, 2, 0.1, True)
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    model = TinyGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=3e-5)
    history = []
    start = time.monotonic()
    for step in range(1, steps + 1):
        if stopped():
            return False
        x, y = batch(tokens["train"], cfg.block_size, 8, rng)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not math.isfinite(loss.item()):
            raise ArithmeticError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 25 == 0 or step == steps:
            model.eval()
            training = aggregate(model, tokens["train"])
            validation = aggregate(model, tokens["validation"])
            point = dict(step=step, train=training, validation=validation,
                         perplexity=math.exp(validation), elapsed=time.monotonic() - start)
            history.append(point)
            progress(point)
            model.train()
    write_checkpoint(path, dict(config=asdict(cfg), weights=model.state_dict(),
                                vocabulary=tokenizer.vocabulary, history=history, splits=counts,
                                steps=steps, seed=seed, tokens_processed=steps * 8 * cfg.block_size))
    return True


class Loaded:
    def __init__(self, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self.model = TinyGPT(ModelConfig(**saved["config"])).eval()
        self.model.load_state_dict(saved["weights"])
        self.char = "vocabulary" in saved
        self.tokenizer = (Characters(saved["vocabulary"]) if self.char else
                          spm.SentencePieceProcessor(model_proto=saved["tokenizer"]))
        self.info = dict(config=saved["config"], parameters=sum(p.numel() for p in self.model.parameters()),
                         bytes=path.stat().st_size, history=saved.get("history", []),
                         steps=saved.get("steps"), seed=saved.get("seed"), splits=saved.get("splits"), device="CPU",
                         tokens_processed=saved.get("tokens_processed"),
                         tokenizer="Character" if self.char else "SentencePiece BPE")

    @torch.inference_mode()
    def complete(self, prompt, count, temperature, seed):
        prefix = self.tokenizer.encode(prompt)
        if not prefix:
            raise ValueError("Enter a prompt with at least one token.")
        ids = torch.tensor([prefix[-self.model.config.block_size:]])
        rng = torch.Generator().manual_seed(seed)
        generated = []
        for _ in range(count):
            logits, _ = self.model(ids[:, -self.model.config.block_size:])
            scores = logits[0, -1].clone()
            scores[0] = -torch.inf
            if not self.char:
                for special in [self.tokenizer.bos_id(), self.tokenizer.eos_id()]:
                    if special >= 0:
                        scores[special] = -torch.inf
            if temperature == 0:
                idx = scores.argmax().item()
            else:
                idx = torch.multinomial((scores / temperature).softmax(-1), 1, generator=rng).item()
            if self.char and idx == 1:
                break
            generated.append(idx)
            ids = torch.cat([ids, torch.tensor([[idx]])], dim=1)
        return self.tokenizer.decode(generated), len(generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=Path, default=ROOT / "bundled/kaggle.pt")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("Steps must be positive")
    torch.set_num_threads(2)
    train(args.output, args.steps, progress=lambda point: print(point, flush=True))
