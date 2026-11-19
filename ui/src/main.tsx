import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowRight,
  AudioLines,
  Blocks,
  BookOpen,
  Bot,
  ChevronRight,
  CircleDot,
  Database,
  Fingerprint,
  FlaskConical,
  History,
  Layers3,
  LoaderCircle,
  MessageSquare,
  Play,
  ScanFace,
  Send,
  ShieldCheck,
  Square,
  Video,
  Workflow,
  X,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./style.css";
import Room from "./Room";

type Page =
  | "Guided room"
  | "Testing lab"
  | "Agent console"
  | "Model playground"
  | "Training studio"
  | "Run history"
  | "Video guide";
type Key = "membership" | "inversion" | "memory";
type Name = "kaggle" | "demo";
type Point = {
  step: number;
  train: number;
  validation: number;
  perplexity: number;
  elapsed: number;
};
type Info = {
  parameters: number;
  config: {
    block_size: number;
    embedding_dim: number;
    n_heads: number;
    n_layers: number;
    vocab_size: number;
  };
  steps: number | null;
  tokenizer: string;
  history: Point[];
  splits: Record<string, number> | null;
  tokens_processed: number | null;
};
type Run = {
  id: number;
  scenario: Key;
  status: string;
  evidence: [string, string][];
  note: string;
  at: string;
  checkpoint_steps: number | null;
};
type Job = {
  status: string;
  total: number;
  step: number;
  history: Point[];
  error?: string;
};
type Turn = {
  message: string;
  source: string;
  reply: string;
  trace: { tool: string; result: string }[];
  at: string;
};
const cases = {
  membership: {
    name: "Membership inference",
    tag: "Training privacy",
    icon: Fingerprint,
    desc: "Compare aggregate loss across training and held-out text.",
    color: "plum",
  },
  inversion: {
    name: "Model inversion",
    tag: "Output privacy",
    icon: ScanFace,
    desc: "Inspect how an output filter handles a synthetic identifier.",
    color: "blue",
  },
  memory: {
    name: "Memory poisoning",
    tag: "Agent integrity",
    icon: Database,
    desc: "Test which sources can change a stored agent preference.",
    color: "green",
  },
};
const keys = Object.keys(cases) as Key[];
const nav = [
  {
    group: "Workspace",
    items: [
      ["Guided room", BookOpen],
      ["Testing lab", FlaskConical],
      ["Agent console", Bot],
      ["Model playground", MessageSquare],
      ["Training studio", AudioLines],
    ],
  },
  {
    group: "Resources",
    items: [
      ["Run history", History],
      ["Video guide", Video],
    ],
  },
] as const;
const titles: Record<Name, string> = {
  kaggle: "MiniGPT · Kaggle",
  demo: "TinyGPT · BPE demo",
};
const video = "https://youtu.be/1WJdmvMoWgE";
async function api<T>(path: string, payload?: unknown): Promise<T> {
  const res = await fetch(
    "/api/" + path,
    payload === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
  );
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "Request failed (" + res.status + ").",
    );
  }
  return res.json();
}
function emblem(Icon: typeof Bot, color: string) {
  return (
    <span className={"emblem " + color}>
      <Icon size={23} strokeWidth={1.7} />
    </span>
  );
}
function Btn({
  children,
  onClick,
  disabled,
  kind = "",
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  kind?: string;
}) {
  return (
    <button className={"btn " + kind} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}
function Chart({ points }: { points: Point[] }) {
  return (
    <div className="plot">
      <ResponsiveContainer width="100%" height={245}>
        <LineChart data={points} margin={{ left: -18, right: 12, top: 12 }}>
          <CartesianGrid
            strokeDasharray="3 5"
            vertical={false}
            stroke="#e5e8ec"
          />
          <XAxis dataKey="step" tick={{ fontSize: 11 }} />
          <YAxis domain={["auto", "auto"]} tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend />
          <Line
            name="Training loss"
            type="monotone"
            dataKey="train"
            stroke="#19826b"
            strokeWidth={2.5}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            name="Validation loss"
            dataKey="validation"
            stroke="#9871ca"
            strokeWidth={2.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
function pageFromHash(): Page {
  const pages = nav.flatMap((g) => g.items.map(([name]) => name));
  return (
    pages.find(
      (name) =>
        name.toLowerCase().replaceAll(" ", "-") === location.hash.slice(1),
    ) || "Guided room"
  );
}
function App() {
  const [page, go] = useState<Page>(pageFromHash),
    [models, setModels] = useState<Record<Name, Info> | null>(null),
    [runs, setRuns] = useState<Run[]>([]),
    [active, setActive] = useState<Run | null>(null);
  const [busy, setBusy] = useState(""),
    [err, error] = useState(""),
    [menu, setMenu] = useState(false),
    [model, choose] = useState<Name>("kaggle"),
    [input, edit] = useState(""),
    [answer, setAnswer] = useState(""),
    [resultTokens, setResultTokens] = useState(0),
    [temp, setTemp] = useState(0.8),
    [limit, setLimit] = useState(80),
    [seed, setSeed] = useState(42);
  const [turns, setTurns] = useState<Turn[]>([]),
    [task, setTask] = useState(""),
    [source, setSource] = useState<"operator" | "document">("operator"),
    [job, setJob] = useState<Job | null>(null),
    [steps, setSteps] = useState(100),
    [videoOn, watch] = useState(false);
  const [session] = useState(() => {
    const id = localStorage.getItem("minigpt.session") || crypto.randomUUID();
    localStorage.setItem("minigpt.session", id);
    return id;
  });
  const locked = useRef(false);
  const navigate = (p: Page) => {
    location.hash = p.toLowerCase().replaceAll(" ", "-");
    go(p);
    setMenu(false);
    error("");
    setActive(null);
    window.scrollTo(0, 0);
  };
  useEffect(() => {
    const sync = () => go(pageFromHash());
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);
  async function action(label: string, fn: () => Promise<void>) {
    if (locked.current) return;
    locked.current = true;
    setBusy(label);
    error("");
    try {
      await fn();
    } catch (e) {
      error(String((e as Error).message));
    } finally {
      locked.current = false;
      setBusy("");
    }
  }
  async function refresh() {
    const [m, r, j, t] = await Promise.all([
      api<Record<Name, Info>>("model/info"),
      api<Run[]>("lab/history"),
      api<Job>("training/status"),
      api<Turn[]>("agent/" + session),
    ]);
    setModels(m);
    setRuns(r);
    setJob(j);
    setTurns(t);
  }
  useEffect(() => {
    refresh().catch((e) => error(e.message));
  }, []);
  useEffect(() => {
    if (!job || !["running", "stopping"].includes(job.status)) return;
    let cancelled = false;
    const timer = setInterval(
      () =>
        api<Job>("training/status")
          .then(async (j) => {
            if (cancelled) return;
            setJob(j);
            if (j.status === "completed") setModels(await api("model/info"));
          })
          .catch((e) => error(e.message)),
      1000,
    );
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [job?.status]);
  useEffect(() => {
    const close = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setActive(null);
        setMenu(false);
      }
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, []);
  async function run(which: Key | "all") {
    await action(which, async () => {
      const result = await api<Run[]>("lab/run", { scenario: which });
      setRuns(await api("lab/history"));
      if (which === "all") navigate("Run history");
      else setActive(result[0]);
    });
  }
  function agent(message = task) {
    if (!message.trim()) return;
    action("agent", async () => {
      const turn = await api<Turn>("agent", {
        session,
        message,
        model,
        source,
      });
      setTurns((v) => [...v, turn]);
      setTask("");
      setRuns(await api("lab/history"));
    });
  }
  const info = models?.[model],
    primary = models?.kaggle,
    training = job?.status === "running" || job?.status === "stopping";
  const pts = job?.history.length ? job.history : primary?.history || [],
    last = pts.at(-1);
  function modelPicker() {
    return (
      <label>
        Checkpoint
        <select
          value={model}
          onChange={(e) => {
            choose(e.target.value as Name);
            setAnswer("");
          }}
        >
          {Object.entries(titles).map(([k, v]) => (
            <option key={k} value={k}>
              {v}
            </option>
          ))}
        </select>
      </label>
    );
  }
  function historyTable(list: Run[]) {
    return (
      <div className="panel history">
        {!list.length ? (
          <div className="empty">
            <History />
            <div>
              <b>No runs yet</b>
              <p>Run a scenario to see its evidence here.</p>
            </div>
          </div>
        ) : (
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>Scenario</th>
                  <th>Result</th>
                  <th>Run time</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {list.map((r) => (
                  <tr key={r.id}>
                    <td>
                      <b>{cases[r.scenario].name}</b>
                    </td>
                    <td>
                      <span
                        className={
                          "pill " + (r.status === "Failed" ? "bad" : "")
                        }
                      >
                        {r.status === "Measured" ? (
                          <Activity size={13} />
                        ) : (
                          <CircleDot size={13} />
                        )}{" "}
                        {r.status}
                      </span>
                    </td>
                    <td>{new Date(r.at).toLocaleString()}</td>
                    <td>
                      <button className="textbtn" onClick={() => setActive(r)}>
                        Evidence <ChevronRight size={14} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    );
  }
  if (page === "Guided room")
    return <Room session={session} exit={() => navigate("Testing lab")} />;
  return (
    <div className="shell">
      <aside className={menu ? "side opened" : "side"}>
        <div className="brand">
          <div className="brandicon">
            <Layers3 size={24} />
          </div>
          <div>
            MiniGPT<small>Model laboratory</small>
          </div>
        </div>
        {nav.map((g) => (
          <div className="navgroup" key={g.group}>
            <div className="navlabel">{g.group}</div>
            {g.items.map(([name, Icon]) => (
              <button
                key={name}
                className={page === name ? "nav sel" : "nav"}
                onClick={() => navigate(name)}
              >
                <Icon size={18} />
                {name}
                {page === name && <i />}
              </button>
            ))}
          </div>
        ))}
        <div className="runtime">
          <span className="dot" /> {models ? "Runtime online" : "Connecting"}
          <small>Local workspace · CPU</small>
        </div>
      </aside>
      {menu && (
        <button
          aria-label="Close navigation"
          className="scrim"
          onClick={() => setMenu(false)}
        />
      )}
      <div className="body">
        <header>
          <button
            className="mobile"
            onClick={() => setMenu(!menu)}
            aria-label="Navigation"
          >
            <Layers3 />
          </button>
          <Blocks size={16} />
          <span>Workspace</span>
          <ChevronRight size={14} />
          <strong>{page}</strong>
          <div className="host">
            <span className={models ? "dot" : "dot offline"} />{" "}
            {models ? "Local engine" : "Connecting"}
          </div>
        </header>
        <main>
          {err && (
            <div role="alert" className="alert">
              {err}
              <button onClick={() => action("refresh", refresh)}>
                Reconnect
              </button>
            </div>
          )}
          {page === "Testing lab" && (
            <>
              <div className="headline">
                <div>
                  <div className="eyeline">
                    <FlaskConical size={15} /> Security workspace
                  </div>
                  <h1>Agent testing lab</h1>
                  <p>
                    Inspect model privacy and agent memory with local,
                    repeatable tests.
                  </p>
                </div>
              </div>
              <section className="hero">
                <div className="hero-content">
                  <div className="hero-tag">
                    <span className="dot" /> Local test suite
                  </div>
                  <h2>LLM security checks</h2>
                  <p>
                    Run three exercises on the Kaggle checkpoint and agent
                    controls, then inspect the evidence from each run.
                  </p>
                  <div className="actions">
                    <Btn
                      kind="mint"
                      disabled={!!busy || !models}
                      onClick={() => run("all")}
                    >
                      {busy === "all" ? (
                        <LoaderCircle className="spin" />
                      ) : (
                        <Play size={15} fill="currentColor" />
                      )}{" "}
                      Run all checks
                    </Btn>
                    <Btn
                      kind="ghostlight"
                      onClick={() => navigate("Video guide")}
                    >
                      <Video size={16} /> Video guide
                    </Btn>
                  </div>
                </div>
                <div className="visual" aria-hidden="true">
                  <i />
                  <i />
                  <i />
                  <div className="axis a" />
                  <div className="axis b" />
                  <span className="sat one">
                    <Fingerprint />
                  </span>
                  <span className="sat two">
                    <Database />
                  </span>
                  <span className="sat three">
                    <ScanFace />
                  </span>
                  <div className="core">
                    <ShieldCheck size={42} strokeWidth={1.35} />
                  </div>
                </div>
              </section>
              <div className="sectionhead">
                <h2>Test scenarios</h2>
                <span>3 exercises</span>
              </div>
              <div className="cards">
                {keys.map((k) => {
                  const c = cases[k];
                  return (
                    <article className="panel card" key={k}>
                      {emblem(c.icon, c.color)}
                      <small>{c.tag}</small>
                      <h3>{c.name}</h3>
                      <p>{c.desc}</p>
                      <button
                        className="cardfoot"
                        disabled={!!busy || !models}
                        onClick={() => run(k)}
                      >
                        {busy === k ? "Running…" : "Run scenario"}
                        <ArrowRight size={16} />
                      </button>
                    </article>
                  );
                })}
              </div>
              <div className="sectionhead">
                <h2>Recent runs</h2>
                <button
                  className="textbtn"
                  onClick={() => navigate("Run history")}
                >
                  View history <ArrowRight size={14} />
                </button>
              </div>
              {historyTable(runs.slice(0, 3))}
            </>
          )}
          {page === "Run history" && (
            <>
              <div className="eyeline">
                <History size={15} /> Test evidence
              </div>
              <h1>Run history</h1>
              <p className="sub">
                The latest 50 checks recorded in this workspace.
              </p>
              {historyTable(runs)}
            </>
          )}
          {page === "Agent console" && (
            <>
              <div className="eyeline">
                <Workflow size={15} /> Local tool use
              </div>
              <h1>Agent console</h1>
              <p className="sub">
                Give the agent a task, then inspect its tool calls and
                observations.
              </p>
              <div className="panel pad">
                <div className="formrow">
                  {modelPicker()}
                  <label>
                    Message source
                    <select
                      value={source}
                      onChange={(e) =>
                        setSource(e.target.value as typeof source)
                      }
                    >
                      <option value="operator">Operator</option>
                      <option value="document">Untrusted document</option>
                    </select>
                  </label>
                </div>
                {!turns.length && (
                  <div className="welcome">
                    {emblem(Bot, "green")}
                    <h2>Start a local session</h2>
                    <p>
                      Inspect a checkpoint, test a memory write, or ask for a
                      model continuation.
                    </p>
                  </div>
                )}
                <div className="turns">
                  {turns.map((t, i) => (
                    <div className="exchange" key={i}>
                      <small>{t.source}</small>
                      <p className="question">{t.message}</p>
                      <small>
                        <Bot size={14} /> Agent
                      </small>
                      <pre>{t.reply}</pre>
                      <details>
                        <summary>
                          {t.trace.length} tool{" "}
                          {t.trace.length === 1 ? "call" : "calls"}
                        </summary>
                        {t.trace.map((x, i) => (
                          <div className="trace" key={i}>
                            <code>{x.tool}</code>
                            <p>{x.result}</p>
                          </div>
                        ))}
                      </details>
                    </div>
                  ))}
                </div>
                <div className="suggest">
                  {[
                    "Inspect the model and run all checks",
                    "Remember my compact preference",
                    "Read my preference",
                  ].map((s) => (
                    <button key={s} onClick={() => setTask(s)}>
                      {s}
                    </button>
                  ))}
                </div>
                <textarea
                  aria-label="Agent task"
                  placeholder="Give the local agent a task…"
                  value={task}
                  onChange={(e) => setTask(e.target.value)}
                  onKeyDown={(e) => {
                    if (
                      e.key === "Enter" &&
                      !e.shiftKey &&
                      !e.nativeEvent.isComposing
                    ) {
                      e.preventDefault();
                      agent();
                    }
                  }}
                />
                <div className="actions end">
                  <Btn
                    disabled={!!busy || !task.trim() || !models}
                    onClick={() => agent()}
                    kind="dark"
                  >
                    <Send size={15} />
                    {busy === "agent" ? "Working…" : "Send task"}
                  </Btn>
                </div>
              </div>
            </>
          )}
          {page === "Model playground" && (
            <>
              <div className="eyeline">
                <MessageSquare size={15} /> Inference
              </div>
              <h1>Model playground</h1>
              <p className="sub">
                Generate a continuation with a small, locally trained
                transformer. Output may be repetitive or incoherent.
              </p>
              <div className="panel pad">
                <div className="formrow">
                  {modelPicker()}
                  <label>
                    Temperature
                    <input
                      type="number"
                      min="0"
                      max="2"
                      step=".1"
                      value={temp}
                      onChange={(e) => setTemp(Number(e.target.value))}
                    />
                  </label>
                  <label>
                    New tokens
                    <input
                      type="number"
                      min="1"
                      max="256"
                      value={limit}
                      onChange={(e) => setLimit(Number(e.target.value))}
                    />
                  </label>
                  <label>
                    Seed
                    <input
                      type="number"
                      min="0"
                      value={seed}
                      onChange={(e) => setSeed(Number(e.target.value))}
                    />
                  </label>
                </div>
                {info && (
                  <p className="fine">
                    {info.parameters.toLocaleString()} parameters ·{" "}
                    {info.config.block_size}-token context · {info.tokenizer}
                  </p>
                )}
                <label>
                  Starting prompt
                  <textarea
                    value={input}
                    onChange={(e) => edit(e.target.value)}
                    placeholder={
                      model === "demo"
                        ? "Hello"
                        : "Question: What is a neural network?"
                    }
                  />
                </label>
                <Btn
                  kind="dark"
                  disabled={!!busy || !input.trim() || !models}
                  onClick={() =>
                    action("generate", async () => {
                      const r = await api<{ text: string; tokens: number }>(
                        "model/generate",
                        {
                          model,
                          prompt: input,
                          tokens: limit,
                          temperature: temp,
                          seed,
                        },
                      );
                      setAnswer(r.text || "(End-of-document token)");
                      setResultTokens(r.tokens);
                    })
                  }
                >
                  <Play size={15} />
                  {busy === "generate" ? "Generating…" : "Generate text"}
                </Btn>
                {answer && (
                  <section className="result">
                    <small>
                      Generated continuation · {resultTokens} tokens
                    </small>
                    <pre>{answer}</pre>
                  </section>
                )}
              </div>
            </>
          )}
          {page === "Training studio" && (
            <>
              <div className="eyeline">
                <AudioLines size={15} /> Local pretraining
              </div>
              <h1>Training studio</h1>
              <p className="sub">
                Train a fresh character model on the Kaggle Q&A corpus. A
                completed run becomes the active Kaggle checkpoint.
              </p>
              <div className="statgrid">
                {[
                  [
                    "Documents",
                    primary?.splits
                      ? Object.values(primary.splits)
                          .reduce((a, b) => a + b, 0)
                          .toLocaleString()
                      : "—",
                  ],
                  ["Parameters", primary?.parameters.toLocaleString() || "—"],
                  ["Validation loss", last?.validation.toFixed(3) || "—"],
                  ["Perplexity", last?.perplexity.toFixed(2) || "—"],
                ].map(([k, v]) => (
                  <div className="panel metric" key={k}>
                    <small>{k}</small>
                    <strong>{v}</strong>
                  </div>
                ))}
              </div>
              <section className="panel pad">
                <h3>Training controls</h3>
                <p className="fine">
                  Fresh random weights · CPU · batch 8 · 128 character tokens ·
                  seed 42
                </p>
                <div className="inline">
                  <label>
                    Training steps
                    <input
                      type="number"
                      min="1"
                      max="1000"
                      value={steps}
                      onChange={(e) => setSteps(Number(e.target.value))}
                    />
                  </label>
                  <Btn
                    kind="dark"
                    disabled={!!busy || training || !models}
                    onClick={() =>
                      action("train", async () => {
                        setJob(
                          await api("training/start", { steps, seed: 42 }),
                        );
                      })
                    }
                  >
                    <Play size={15} /> Start training
                  </Btn>
                  {training && (
                    <Btn
                      onClick={() =>
                        action("stop", async () =>
                          setJob(await api("training/stop", {})),
                        )
                      }
                    >
                      <Square size={14} /> Stop
                    </Btn>
                  )}
                </div>
                <p className="fine" role="status">
                  {job?.status === "idle"
                    ? "Loaded checkpoint · " +
                      (primary?.steps || "—") +
                      " steps"
                    : job?.status +
                      " · " +
                      job?.step +
                      " / " +
                      job?.total +
                      " steps"}{" "}
                  {job?.error}
                </p>
                {training && <progress max={job?.total} value={job?.step} />}
              </section>
              <div className="sectionhead">
                <h2>Learning curve</h2>
                <span>Cross-entropy · nats</span>
              </div>
              <div className="panel pad">
                {pts.length ? (
                  <Chart points={pts} />
                ) : (
                  <p>Waiting for the first measurement.</p>
                )}
              </div>
            </>
          )}
          {page === "Video guide" && (
            <>
              <div className="eyeline">
                <BookOpen size={15} /> Reference
              </div>
              <h1>Video guide</h1>
              <p className="sub">
                LLM Attacks Explained: Membership Inference, Model Inversion
                &amp; Memory Poisoning.
              </p>
              <div className="player">
                {videoOn ? (
                  <iframe
                    src="https://www.youtube-nocookie.com/embed/1WJdmvMoWgE?autoplay=1"
                    allow="autoplay; encrypted-media; picture-in-picture"
                    allowFullScreen
                    title="LLM Attacks Explained"
                  />
                ) : (
                  <button onClick={() => watch(true)}>
                    <span>
                      <Play size={32} fill="currentColor" />
                    </span>
                    Load video from YouTube
                  </button>
                )}
              </div>
              <p>
                <a href={video} target="_blank" rel="noreferrer">
                  Open original video on YouTube ↗
                </a>
              </p>
              <div className="sectionhead">
                <h2>Companion exercises</h2>
              </div>
              {keys.map((k) => (
                <div key={k} className="panel guide">
                  {emblem(cases[k].icon, cases[k].color)}
                  <div>
                    <h3>{cases[k].name}</h3>
                    <p>{cases[k].desc}</p>
                  </div>
                  <Btn disabled={!!busy || !models} onClick={() => run(k)}>
                    Run <ArrowRight size={15} />
                  </Btn>
                </div>
              ))}
            </>
          )}
        </main>
      </div>
      {active && (
        <div className="backdrop" onClick={() => setActive(null)}>
          <section
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-label="Run evidence"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              autoFocus
              className="close"
              aria-label="Close evidence"
              onClick={() => setActive(null)}
            >
              <X size={19} />
            </button>
            {emblem(cases[active.scenario].icon, cases[active.scenario].color)}
            <div className="eyeline">
              Run {active.id} · {active.status}
            </div>
            <h2>{cases[active.scenario].name}</h2>
            {active.evidence.map(([k, v]) => (
              <div className="evidence" key={k}>
                <small>{k}</small>
                <strong>{v}</strong>
              </div>
            ))}
            <p>{active.note}</p>
            <small>
              {new Date(active.at).toLocaleString()}
              {active.checkpoint_steps
                ? " · Kaggle checkpoint, " + active.checkpoint_steps + " steps"
                : ""}
            </small>
          </section>
        </div>
      )}
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
