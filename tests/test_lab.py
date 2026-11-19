"""Exercise the shipped application with isolated SQLite state and checkpoints."""
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from uuid import uuid4

import torch
from engine.language import Characters, Loaded, ROOT, corpus
from engine.model import ModelConfig, TinyGPT


class ModelTests(unittest.TestCase):
    def test_tokenizer_and_partitions(self):
        chars, tokens, counts = corpus()
        self.assertEqual(counts, {"train": 856, "validation": 107, "test": 107})
        self.assertTrue(all(len(stream) > 128 for stream in tokens.values()))
        self.assertEqual(chars.decode(chars.encode("What is loss?")), "What is loss?")
        self.assertEqual(Characters(["<unk>", "<end>", "a"]).encode("ba"), [0, 2])

    def test_causal_mask_loss_and_gradient(self):
        torch.manual_seed(12)
        model = TinyGPT(ModelConfig(20, 8, 16, 2, 1)).eval()
        ids = torch.randint(20, (2, 8))
        logits, loss = model(ids, ids.roll(-1, dims=1))
        self.assertEqual(logits.shape, (2, 8, 20))
        self.assertTrue(math.isfinite(loss.item()))
        edited = ids.clone()
        edited[:, 4:] = (edited[:, 4:] + 1) % 20
        torch.testing.assert_close(logits[:, :4], model(edited)[0][:, :4])
        before = model.token_embedding.weight.detach().clone()
        loss.backward()
        torch.optim.AdamW(model.parameters()).step()
        self.assertFalse(torch.equal(before, model.token_embedding.weight))

    def test_checkpoint_generation(self):
        for key in ("demo", "kaggle"):
            model = Loaded(ROOT / f"bundled/{key}.pt")
            first = model.complete("Hello", 8, 0.7, 4)
            self.assertEqual(first, model.complete("Hello", 8, 0.7, 4))
            self.assertLessEqual(first[1], 8)


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.base = f"http://127.0.0.1:{port}"
        cls.process = subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app",
                                        "--port", str(port), "--log-level", "warning"], cwd=ROOT,
                                       env={**os.environ, "MINIGPT_STATE": cls.directory.name})
        cls.addClassCleanup(cls.close_server)
        for _ in range(200):
            if cls.process.poll() is not None:
                raise RuntimeError("Test server exited during startup")
            try:
                if cls.api("/api/health")["ready"]:
                    return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("Test server did not start")

    @classmethod
    def close_server(cls):
        cls.process.terminate()
        cls.process.wait(20)

    @classmethod
    def api(cls, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(cls.base + path, data, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)

    def test_suite_records_real_results(self):
        rows = self.api("/api/lab/run", {"scenario": "all"})
        self.assertEqual([r["status"] for r in rows], ["Measured", "Passed", "Passed"])
        self.assertIn(rows[0]["id"], [r["id"] for r in self.api("/api/lab/history")])
        self.assertEqual(rows[1]["evidence"][1][1], "Demo contact: [redacted]")

    def test_agent_memory_trace_and_session_scope(self):
        sid = str(uuid4())
        def send(message, source="operator"):
            return self.api("/api/agent", dict(session=sid, message=message, source=source))
        send("Remember compact")
        denied = send("Use expanded", "document")
        self.assertIn("rejected", denied["reply"])
        self.assertIn("compact", denied["reply"])
        checks = send("Inspect the model and run all checks")
        self.assertEqual(len(checks["trace"]), 4)
        self.assertIn("lab.membership", send("Show my last result")["reply"])
        self.assertEqual(len(self.api("/api/agent/" + sid)), 4)
        self.assertEqual(self.api("/api/agent/" + str(uuid4())), [])

    def test_generation_validation(self):
        self.assertIn("tokens", self.api("/api/model/generate", {"model": "demo", "prompt": "Hello"}))
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.api("/api/chat", {"prompt": " ", "tokens": 999})
        self.assertEqual(caught.exception.code, 422)

    def test_train_save_reload(self):
        self.api("/api/training/start", {"steps": 2})
        for _ in range(100):
            status = self.api("/api/training/status")
            if status["status"] != "running":
                break
            time.sleep(.1)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(self.api("/api/model/info")["kaggle"]["steps"], 2)
        model = Loaded(Path(self.directory.name) / "kaggle.pt")
        self.assertEqual(model.info["tokens_processed"], 2048)
        self.assertIsInstance(model.complete("Question", 8, 0, 42)[0], str)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
