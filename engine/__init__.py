
"""Local model training, inference, and security exercises for the MiniGPT lab.

To follow a learner's message, start with `engine.room.respond`. It selects
the task handler and returns a reply with evidence for the FastAPI service.

Package guide
-------------
room                Lesson definitions, task routing, token budgets, and grading.
checks              Memory write policy, output redaction, and privacy checks.
language            Corpus preparation, training, checkpoint I/O, and generation.
model               TinyGPT decoder and its configuration.
transformer_blocks  Causal attention heads and residual transformer blocks.

Import the component you need from its module, for example::

    from engine.model import ModelConfig, TinyGPT
"""
