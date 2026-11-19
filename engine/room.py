"""Session-scoped lessons for the local security room."""
import re
from uuid import uuid4

from .checks import SessionMemory, exercise, publish

OUTPUT_LIMIT = 12
BUDGET_BASELINE = f"Generate {OUTPUT_LIMIT} tokens: A small model"
BUDGET_BOUNDARY = f"Generate {OUTPUT_LIMIT + 1} tokens: A small model"

LESSONS = [
    dict(key="intro", title="Meet your model", role="Model operator", topic="Orientation",
         brief="Meet the character-level model behind the playground. Inspect its checkpoint, then try generating a short continuation.",
         goal="Find the context length of the loaded Kaggle checkpoint.",
         question="How many tokens fit in its context window?", prompt="Inspect your checkpoint",
         greeting="I can read the loaded checkpoint and sample its next tokens. What would you like to inspect?"),
    dict(key="privacy", title="Compare the splits", role="Privacy analyst", topic="Membership inference",
         brief="A model can fit its training text more closely than held-out text. Compare two aggregate measurements and decide what this evidence supports.",
         goal="Measure both splits on the same checkpoint and identify the lower loss.",
         question="Which split has lower loss: training, validation, or equal?", prompt="Compare training and validation",
         greeting="I have access to the local training and validation partitions. Ask me to measure their average losses."),
    dict(key="release", title="Review a disclosure", role="Release reviewer", topic="Model inversion",
         brief="Review a synthetic support note before publication. The outgoing filter should remove its test contact while preserving the public case reference.",
         goal="Inspect the released text and identify its public case reference.",
         question="Which case reference survived publication?", prompt="Review a sample note",
         greeting="I can create a fictional support note and show what its recipient would see. You can supply an invented address ending in @example.test."),
    dict(key="memory", title="Trace a memory write", role="Memory custodian", topic="Memory poisoning",
         brief="Give the agent a display preference. Then submit a document that proposes the opposite value. Read the stored preference in a separate turn.",
         goal="Observe an operator write, a conflicting document write, and a read-back in that order.",
         question="Which display preference survived?", prompt="Remember compact display",
         greeting="My store accepts operator preferences. Choose compact or expanded, then try the other value as a document."),
    dict(key="review", title="Verify a dependency", role="Build reviewer", topic="User trust",
         brief="An assistant drafted a package suggestion. Use the lab's installed Python metadata and pinned runtime manifest to verify its version before trusting the suggestion.",
         goal="Choose torch, fastapi, or uvicorn and compare its installed version with the manifest.",
         question="What exact installed version did you verify?", prompt="Verify FastAPI",
         greeting="I can inspect three dependencies in this container: torch, fastapi and uvicorn. Which one should I verify?"),
    dict(key="budget", title="Test a token budget", role="Resource controller", topic="OWASP LLM10:2025 · Unbounded Consumption",
         brief="Every generated token costs compute. This exercise gives each request a small output allowance. Compare two requests that differ by one token.",
         goal="Run the at-limit and over-limit examples. Compare their decisions and model calls.",
         question="How many model calls did the rejected request make?", prompt=BUDGET_BASELINE,
         examples=[BUDGET_BASELINE, BUDGET_BOUNDARY],
         greeting="Choose an example or type “Generate 4 tokens: Hello”. I check the requested output budget before calling the local model."),
]
KEYS = {lesson["key"] for lesson in LESSONS}
EXPLANATIONS = {
    "intro": "Each character is encoded as a token. At generation time the decoder attends to the most recent window and predicts the next character.",
    "privacy": "Membership inference asks whether a particular record was in training. The gap measured in this room describes generalization across groups. It cannot settle an individual record’s membership.",
    "release": "Model inversion concerns reconstruction of private attributes. This task measures a narrower control: whether publication removes a reserved test address and keeps the public reference.",
    "memory": "Memory poisoning can persist misleading information into later turns. The source policy is checked at write time, and reading the saved value independently verifies the result.",
    "review": "A generated dependency suggestion can contain an inaccurate name or version. Installed distribution metadata supplies the runtime version and the manifest supplies its intended pin.",
    "budget": "An output cap puts an upper bound on decoding work for one request. The guard checks the requested maximum before inference. A model that reaches its end token early may produce fewer tokens. Aggregate traffic needs separate controls such as quotas.",
}


def blank():
    return {"complete": False, "answer": None, "evidence": [], "notes": {}, "stage": 0}


def result(text, tool="guide", evidence=()):
    return {"reply": text, "tool": tool, "evidence": list(evidence)}


def normalized(text):
    return " ".join(text.casefold().split()).strip(" .!?")


def budget_trial(message, state, model):
    match = re.fullmatch(r"generate\s+(\d{1,4})\s+tokens?\s*:\s*(.+)", message.strip(), re.I | re.S)
    if not match or int(match[1]) < 1:
        return result("Use a positive token count and some starting text. Try “Generate 4 tokens: Hello”.")
    requested, prefix = int(match[1]), match[2].strip()
    allowed = requested <= OUTPUT_LIMIT
    calls = produced = 0
    if allowed:
        continuation, produced = model.complete(prefix, requested, 0.0, 42)
        calls = 1
        if requested == OUTPUT_LIMIT:
            state["notes"]["at_limit"] = True
            state["notes"]["baseline_tokens"] = produced
    else:
        state["notes"]["denied_calls"] = calls
    evidence = [["Decision", "Allowed" if allowed else "Rejected"], ["Requested maximum", str(requested)],
                ["Output limit", str(OUTPUT_LIMIT)], ["Model calls", str(calls)], ["Tokens generated", str(produced)]]
    if allowed:
        evidence.append(["Generated continuation", publish(continuation) or "(end of document)"])
    state["evidence"] = evidence
    if state["notes"].get("at_limit") and "denied_calls" in state["notes"]:
        state["answer"] = str(state["notes"]["denied_calls"])
    reply = ("The request fits the budget. MiniGPT sampled a continuation." if allowed else
             "The requested output exceeds the budget. The server rejected it before decoding.")
    return result(reply, "budget.check", evidence)


def respond(lesson, state, message, mode, model, splits, packages):
    key = lesson["key"]
    query = normalized(message)
    if mode == "answer":
        if not state["answer"]:
            return result("Collect the evidence for this task first. " + lesson["goal"], "answer.check")
        aliases = {"train": "training", "held-out": "validation", "held out": "validation",
                   "validation split": "validation", "training split": "training", "same": "equal"}
        clean = re.sub(r"\s+tokens?$", "", query) if key == "intro" else query
        if key == "budget":
            clean = re.sub(r"\s+(?:model\s+)?calls?$", "", query)
            clean = "0" if clean in ("zero", "none") else clean
        if (aliases.get(clean, clean) if key == "privacy" else clean) == state["answer"]:
            state["complete"] = True
            return result("Verified against your session evidence. You’ve completed this task.", "answer.check")
        return result("That answer differs from the observation. Review the evidence and try again.", "answer.check")

    if mode == "document" and key != "memory":
        return result("The document channel is available in the memory task.")
    if mode == "chat":
        if key == "budget" and re.match(r"^generate\s", query):
            return budget_trial(message, state, model)
        if query in ("hi", "hello", "hey", "who are you", "what can you do"):
            return result(lesson["greeting"])
        if re.search(r"\b(interpret|what does (that|this) mean|tell me more|explain (that|this|the result))\b", query):
            return result(EXPLANATIONS[key], "evidence.explain", state["evidence"])
        for term, topic in (("membership inference", "privacy"), ("model inversion", "release"),
                            ("memory poisoning", "memory"), ("context window", "intro")):
            if term in query and re.search(r"\b(what is|explain|define|tell me about)\b", query):
                return result(EXPLANATIONS[topic])
        if re.search(r"\b(hint|help|goal|what next|what now)\b", query):
            if key == "budget":
                if not state["notes"].get("at_limit"):
                    return result(f"Start at the limit. Try “{BUDGET_BASELINE}” and inspect the generated continuation.")
                if "denied_calls" not in state["notes"]:
                    return result(f"Now change just the requested maximum. Try “{BUDGET_BOUNDARY}”.")
                return result("Read the Model calls row of the rejected request and submit that number in Your finding.")
            if key == "memory":
                hints = ["Set either compact or expanded as the operator.",
                         "Select Document and propose the opposite preference.",
                         "Switch back to Operator and ask me to read the saved preference."]
                return result(hints[min(state["stage"], 2)])
            return result(lesson["goal"] + " Try “" + lesson["prompt"] + "”.")
        if re.search(r"\b(evidence|last result|what did (we|you) find)\b", query):
            return result("Here is the last observation." if state["evidence"] else "Collect your first observation using the task prompt.",
                          "evidence.read", state["evidence"])
        if "training data" in query or "dataset" in query or "corpus" in query:
            counts = model.info["splits"]
            return result(f"The Kaggle checkpoint uses 1000+ Data Science Concepts. Its character vocabulary was fitted on {counts['train']} training documents. "
                          f"{counts['validation']} documents were reserved for validation and {counts['test']} for testing.", "checkpoint.dataset")
        if re.search(r"\b(why|explain|what is|what does|what can)\b", query) and not re.search(r"\b(version|context|stored|saved|preference|loss)\b",query):
            return result(EXPLANATIONS[key])
    values = []
    answer = None
    if key == "budget":
        return budget_trial(message, state, model)
    if key == "intro" and re.search(r"\b(generate|continue|sample)\b", query):
        prefix = message.partition(":")[2].strip() or "Question: What is learning?"
        text, count = model.complete(prefix, 40, 0.8, 42)
        return result("The local Kaggle model generated this continuation.", "model.generate",
                      [["Prompt", prefix], ["Generated text", publish(text) or "(end of document)"], ["New tokens", str(count)]])
    if key == "intro" and re.search(r"\b(model|inspect|checkpoint|context|demo|demonstration|start)\b", query):
        info = model.info
        answer = str(info["config"]["block_size"])
        values = [["Context window", answer + " tokens"], ["Tokenizer", info["tokenizer"]], ["Parameters", f"{info['parameters']:,}"]]
        reply, tool = "I read these values from the active Kaggle checkpoint.", "checkpoint.inspect"
    elif key == "privacy" and re.search(r"\b(compare|measure|validation|training|split|loss|demo|demonstration)\b", query):
        _, values, reply = exercise("membership", model, splits)
        a, b = [float(value.split()[0]) for _, value in values[:2]]
        answer = "training" if a < b else "validation" if b < a else "equal"
        tool = "privacy.compare"
    elif key == "release" and re.search(r"\b(review|release|publish|note|sample|demo|demonstration)\b|@example\.test", query):
        address = re.search(r"[a-z0-9._%+-]{1,50}@example\.test\b", query)
        case = "CASE-" + uuid4().hex[:6].upper()
        sample = f"{case} | Follow up with {address[0] if address else 'robin@example.test'}"
        values = [["Synthetic draft", sample], ["Published text", publish(sample)]]
        answer, reply, tool = case.lower(), "I ran the note through the playground’s publication filter. Inspect what remains visible.", "output.publish"
    elif key == "memory":
        wanted = re.search(r"\b(compact|expanded)\b", query)
        store = SessionMemory()
        store.notes = state["notes"]
        before = store.notes.get("display")
        if mode == "document":
            if not wanted:
                return result("The fixture accepts a proposed compact or expanded display preference.")
            allowed = store.write("display", wanted[0], "document")
            if state["stage"] == 1 and wanted[0] != before and not allowed:
                state["stage"] = 2
            values = [["Source", "Document"], ["Requested", wanted[0]], ["Decision", "Accepted" if allowed else "Rejected"]]
            reply, tool = "The source check ran. Read the saved preference to verify its effect.", "memory.write"
        elif wanted and re.search(r"\b(remember|set|use|save|prefer)\b", query):
            store.write("display", wanted[0], "operator")
            state.update(stage=1, answer=None)
            values = [["Source", "Operator"], ["Committed preference", wanted[0]]]
            reply, tool = "Saved. Switch the composer to Document and propose the opposite display preference.", "memory.write"
        elif re.search(r"\b(read|recall|stored|saved|preference)\b", query):
            values = [["Stored preference", before or "Unset"]]
            if state["stage"] >= 2:
                answer = before
            reply, tool = "I read the value directly from this room session’s store.", "memory.read"
        else:
            return result(lesson["greeting"])
    elif key == "review" and re.search(r"\b(torch|fastapi|uvicorn)\b", query):
        pkg = re.search(r"\b(torch|fastapi|uvicorn)\b", query)[0]
        installed, pinned = packages(pkg)
        values = [["Distribution", pkg], ["Installed version", installed], ["Manifest pin", pinned],
                  ["Comparison", "Match" if installed == pinned else "Different"]]
        answer, reply, tool = installed.lower(), "I compared the installed distribution with its requirements.txt entry.", "runtime.verify"
    else:
        return result("I can guide this exercise and inspect its evidence. " + lesson["greeting"] + " Try “" + lesson["prompt"] + "”.")
    state["evidence"] = values
    if answer is not None:
        state["answer"] = answer
    return result(reply, tool, values)
