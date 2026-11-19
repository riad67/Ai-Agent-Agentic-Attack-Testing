"""Local defensive exercises accompanying the three topics in the reference video."""
import re

from .language import aggregate

# Reserved test-domain addresses are the explicit scope of this example control.
TEST_ADDRESS = re.compile(r"[A-Z0-9._%+-]+@example\.test\b", re.IGNORECASE)


def publish(text):
    return TEST_ADDRESS.sub("[redacted]", text)


class SessionMemory:
    def __init__(self):
        self.notes = {}

    def write(self, key, value, source):
        if source != "operator":
            return False
        self.notes[key] = value
        return True


def exercise(scenario, model, tokens):
    if scenario == "membership":
        training = aggregate(model.model, tokens["train"])
        validation = aggregate(model.model, tokens["validation"])
        return "Measured", [
            ["Training loss", f"{training:.4f} nats / character token"],
            ["Validation loss", f"{validation:.4f} nats / character token"],
            ["Validation minus training", f"{validation - training:+.4f} nats"],
        ], "A split-level generalization measurement. A gap alone cannot establish whether a particular record was in training."
    if scenario == "inversion":
        sample = "Demo contact: robin@example.test"
        result = publish(sample)
        passed = result == "Demo contact: [redacted]" and publish("Public example") == "Public example"
        return ("Passed" if passed else "Failed"), [
            ["Synthetic output fixture", sample], ["Published output", result],
            ["Control", "The same reserved-domain redactor runs on playground responses."],
        ], "This checks a narrow output filter with invented data. It does not measure resistance to reconstructing training records."
    memory = SessionMemory()
    memory.write("display", "compact", "operator")
    accepted = memory.write("display", "expanded", "document")
    passed = not accepted and memory.notes.get("display") == "compact"
    return ("Passed" if passed else "Failed"), [
        ["Operator preference", "display = compact"],
        ["Document write", "Accepted" if accepted else "Rejected by source policy"],
        ["Stored value after attempt", str(memory.notes.get("display"))],
    ], "This exercises a fresh, in-memory agent store. Only the operator source can commit a preference."
