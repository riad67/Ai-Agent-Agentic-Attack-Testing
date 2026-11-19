import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  Bot,
  Braces,
  Cpu,
  Database,
  Monitor,
  X,
} from "lucide-react";
import "./architecture.css";

type Trace = {
  text: string;
  tool: string;
  evidence: [string, string][];
  task: string;
  mode: string;
  at: string;
};
type Model = {
  parameters: number;
  config: {
    n_layers: number;
    n_heads: number;
    block_size: number;
    embedding_dim: number;
  };
};
export default function Architecture({
  last,
  model,
  close,
}: {
  last?: Trace;
  model?: Model;
  close: () => void;
}) {
  const [step, setStep] = useState(0);
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const el = dialog.current;
    if (el && !el.open) el.showModal();
    return () => el?.close();
  }, []);
  const calls = last?.evidence.find(([key]) => key === "Model calls")?.[1];
  const runtime =
    last?.tool === "budget.check"
      ? calls === "0"
        ? "Budget rejected · decoder skipped"
        : "Budget accepted · local inference ran"
      : last?.tool;
  const stages = [
    {
      name: "Interface",
      icon: Monitor,
      part: "React · browser",
      body: "The room sends the selected task, message source, text and a unique request ID. Your browser keeps the room session ID so you can return to your findings.",
      value: last?.text,
    },
    {
      name: "API",
      icon: Braces,
      part: "FastAPI · request validation",
      body: "FastAPI validates the task, source and text length. A repeated request ID reuses its recorded result so a retry cannot run the same turn twice.",
      value: last ? "Accepted room turn · " + last.mode : undefined,
    },
    {
      name: "Router",
      icon: Bot,
      part: "Python · task and source policy",
      body: "The task router chooses the handler for this exercise. Hints, answer checks, measurements and generation take different paths. A document message can propose a memory write inside the memory exercise.",
      value: last?.task,
    },
    {
      name: "Execution",
      icon: Cpu,
      part: "Local tools · PyTorch on CPU",
      body: "Inspection reads local metadata. Generation calls the character-level decoder. The LLM10 exercise checks its output allowance before decoding. The model predicts one token at a time until it reaches its limit or an end token.",
      value: runtime,
    },
    {
      name: "Storage",
      icon: Database,
      part: "SQLite · session and task",
      body: "The server commits the tool result, evidence and updated task state together. The interface displays that record, and grading uses the observations collected in your session.",
      value: last ? new Date(last.at).toLocaleString() : undefined,
    },
  ];
  const selected = stages[step],
    Icon = selected.icon;
  return (
    <dialog
      ref={dialog}
      className="architecture"
      aria-labelledby="architecture-heading"
      onCancel={close}
    >
      <div className="arch-head">
        <div>
          <h2 id="architecture-heading">Agent architecture</h2>
          <p>Follow a room message from the browser to its saved result.</p>
        </div>
        <button aria-label="Close architecture" onClick={close}>
          <X size={19} />
        </button>
      </div>
      <div className="arch-nav" role="group" aria-label="Request path">
        {stages.map((s, i) => (
          <button
            key={s.name}
            aria-pressed={i === step}
            onClick={() => setStep(i)}
          >
            <s.icon size={19} />
            {s.name}
          </button>
        ))}
      </div>
      <section className="arch-detail" aria-live="polite">
        <div className="arch-component">
          <Icon size={19} />
          {selected.part}
        </div>
        <p>{selected.body}</p>
        {selected.value && (
          <div className="arch-observation">
            <small>Latest turn in this task</small>
            <code>{selected.value}</code>
          </div>
        )}
      </section>
      {model && (
        <div className="arch-model">
          <strong>Loaded MiniGPT</strong>
          <span>
            {model.parameters.toLocaleString()} parameters ·{" "}
            {model.config.n_layers} blocks · {model.config.n_heads} heads ·{" "}
            {model.config.embedding_dim} embedding width ·{" "}
            {model.config.block_size}-token context
          </span>
        </div>
      )}
      <div className="arch-footer">
        <span>
          Component {step + 1} of {stages.length}
        </span>
        <button
          onClick={() =>
            step === stages.length - 1 ? close() : setStep(step + 1)
          }
        >
          {step === stages.length - 1 ? "Return to room" : "Next component"}{" "}
          <ArrowRight size={14} />
        </button>
      </div>
    </dialog>
  );
}
