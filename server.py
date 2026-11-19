import json
import os
import sqlite3
import threading
import re
from uuid import UUID
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine.checks import SessionMemory, exercise, publish
from engine.language import ROOT, Loaded, corpus, train

STATE = Path(os.environ.get("MINIGPT_STATE", ROOT / "state"))
SCENARIOS = ("membership", "inversion", "memory")
Case = Literal["membership", "inversion", "memory", "all"]
ModelName = Literal["kaggle", "demo"]
lock = threading.RLock()
stop_event = threading.Event()
models = {}
splits = {}
job = {"status": "idle", "step": 0, "total": 0, "history": []}


def connect():
    return sqlite3.connect(STATE / "lab.sqlite3", timeout=10)


@asynccontextmanager
async def lifespan(app):
    STATE.mkdir(parents=True, exist_ok=True)
    with closing(connect()) as db:
        db.execute("CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY, session TEXT, payload TEXT)")
        db.execute("CREATE INDEX IF NOT EXISTS turns_session ON turns(session, id)")
        db.execute("CREATE TABLE IF NOT EXISTS preferences (session TEXT PRIMARY KEY, value TEXT)")
        db.commit()
    torch.set_num_threads(2)
    latest = STATE / "kaggle.pt"
    models["kaggle"] = Loaded(latest if latest.exists() else ROOT / "bundled/kaggle.pt")
    models["demo"] = Loaded(ROOT / "bundled/demo.pt")
    _, data, _ = corpus()
    splits.update(data)
    yield
    stop_event.set()


app = FastAPI(title="MiniGPT laboratory", lifespan=lifespan)


class Prompt(BaseModel):
    model: ModelName = "kaggle"
    prompt: str = Field(min_length=1, max_length=3000)
    tokens: int = Field(default=80, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0, le=2)
    seed: int = Field(default=42, ge=0, le=2147483647)


class Run(BaseModel):
    scenario: Case


class Training(BaseModel):
    steps: int = Field(default=100, ge=1, le=1000)
    seed: int = Field(default=42, ge=0, le=2147483647)


@app.get("/api/health")
def health():
    return {"ready": len(models) == 2}


@app.get("/api/model/info")
def model_info():
    with lock:
        return {key: value.info for key, value in models.items()}


@app.post("/api/chat")
@app.post("/api/model/generate")
def generate(body: Prompt):
    if not body.prompt.strip():
        raise HTTPException(422, "Enter a prompt containing text.")
    with lock:
        model = models[body.model]
    try:
        result, count = model.complete(body.prompt, body.tokens, body.temperature, body.seed)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return {"text": publish(result), "tokens": count, "model": body.model}


@app.post("/api/lab/run")
def lab_run(body: Run):
    cases = SCENARIOS if body.scenario == "all" else [body.scenario]
    with lock:
        model = models["kaggle"]
    results = []
    with closing(connect()) as db, db:
        for name in cases:
            status, evidence, note = exercise(name, model, splits)
            result = {"scenario": name, "status": status, "evidence": evidence, "note": note,
                      "at": datetime.now(timezone.utc).isoformat(),
                      "checkpoint_steps": model.info["steps"] if name == "membership" else None,
                      "checkpoint_seed": model.info.get("seed") if name == "membership" else None}
            row = db.execute("INSERT INTO runs(payload) VALUES (?)", (json.dumps(result),))
            results.append(dict(result, id=row.lastrowid))
    return results


@app.get("/api/lab/history")
def history():
    with closing(connect()) as db:
        records = db.execute("SELECT id,payload FROM runs ORDER BY id DESC LIMIT 50").fetchall()
    return [dict(json.loads(data), id=key) for key, data in records]


@app.get("/api/training/status")
def training_status():
    with lock:
        return {**job, "history": list(job["history"])}


def worker(steps, seed):
    def report(point):
        with lock:
            job["step"] = point["step"]
            job["history"].append(point)
    try:
        path = STATE / "kaggle.pt"
        complete = train(path, steps, seed, report, stop_event.is_set)
        if complete:
            loaded = Loaded(path)
        with lock:
            if complete:
                models["kaggle"] = loaded
            job["status"] = "completed" if complete else "stopped"
    except Exception as error:
        with lock:
            job.update(status="failed", error=str(error))


@app.post("/api/training/start", status_code=202)
def start_training(body: Training):
    with lock:
        if job["status"] in ("running", "stopping"):
            raise HTTPException(409, "A training run is already active.")
        job.clear()
        job.update(status="running", step=0, total=body.steps, history=[])
        stop_event.clear()
        threading.Thread(target=worker, args=(body.steps, body.seed), daemon=True).start()
        return training_status()


@app.post("/api/training/stop")
def stop_training():
    with lock:
        if job["status"] == "running":
            job["status"] = "stopping"
            stop_event.set()
        return training_status()


class AgentRequest(BaseModel):
    session: UUID
    message: str = Field(min_length=1, max_length=2000)
    model: ModelName = "kaggle"
    source: Literal["operator", "document"] = "operator"


@app.get("/api/agent/{session}")
def agent_history(session: UUID):
    with closing(connect()) as db:
        rows = db.execute("SELECT payload FROM turns WHERE session=? ORDER BY id DESC LIMIT 50",
                          (str(session),)).fetchall()
    return [json.loads(row[0]) for row in reversed(rows)]


@app.post("/api/agent")
def agent(body: AgentRequest):
    message = body.message.strip()
    if not message:
        raise HTTPException(422, "Enter a task for the agent.")
    sid = str(body.session)
    query = message.casefold()
    traces = []
    outputs = []
    def record(tool, result):
        traces.append({"tool": tool, "result": publish(result)})
        outputs.append(publish(result))

    preference = re.search(r"\b(compact|expanded)\b", query)
    write_requested = preference and re.search(r"\b(remember|save|set|use|prefer|switch)\b", query)
    if body.source == "document" or write_requested or "preference" in query:
        with closing(connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT value FROM preferences WHERE session=?", (sid,)).fetchone()
            memory = SessionMemory()
            if row:
                memory.notes["display"] = row[0]
            if write_requested or body.source == "document":
                accepted = memory.write("display", preference[1] if preference else "", body.source)
                if accepted:
                    db.execute("INSERT OR REPLACE INTO preferences VALUES (?,?)",
                               (sid, memory.notes["display"]))
                record("memory.write", "Operator preference saved." if accepted
                       else "Document write rejected by the source policy.")
            record("memory.read", "Display preference: " + memory.notes.get("display", "unset"))
    if body.source == "operator":
        if any(phrase in query for phrase in ("last check", "last run", "last result")):
            prior = agent_history(body.session)
            prior_checks = next(([call for call in turn["trace"] if call["tool"].startswith("lab.")]
                                 for turn in reversed(prior)
                                 if any(call["tool"].startswith("lab.") for call in turn["trace"])), [])
            record("history.read", "\n".join(f"{call['tool']} · {call['result']}" for call in prior_checks)
                   or "This session has no previous agent checks.")
        if any(word in query for word in ("inspect", "model info", "architecture")):
            info = models[body.model].info
            record("model.inspect", f"{body.model} · {info['parameters']:,} parameters · "
                   f"{info['config']['block_size']} tokens of context · {info['tokenizer']} · CPU")
        chosen = [name for name in SCENARIOS if name in query]
        if "poison" in query and "memory" not in chosen:
            chosen.append("memory")
        if "suite" in query or "all checks" in query:
            chosen = list(SCENARIOS)
        for name in chosen:
            result = lab_run(Run(scenario=name))[0]
            details = " · ".join(f"{k}: {v}" for k, v in result["evidence"])
            record("lab." + name, f"{result['status']} | {details}")
        if not traces:
            reply = generate(Prompt(model=body.model, prompt=message, tokens=96))
            record("model.generate", reply["text"] or "(End-of-document token)")
    turn = {"message": message, "source": body.source, "model": body.model,
            "reply": "\n\n".join(outputs), "trace": traces,
            "at": datetime.now(timezone.utc).isoformat()}
    with closing(connect()) as db, db:
        db.execute("INSERT INTO turns(session,payload) VALUES (?,?)", (sid, json.dumps(turn)))
    return turn


app.mount("/", StaticFiles(directory=ROOT / "ui/dist", html=True), name="ui")
