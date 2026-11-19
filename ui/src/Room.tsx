import { useEffect, useRef, useState } from "react";
import {
  ArrowDownLeft,
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Bot,
  Check,
  CheckCheck,
  ChevronDown,
  ChevronRight,
  FlaskConical,
  GraduationCap,
  Layers3,
  LoaderCircle,
  LockKeyhole,
  Send,
  Workflow,
  Video,
} from "lucide-react";
import "./room.css";
import Architecture from "./Architecture";

type Key = "intro" | "privacy" | "release" | "memory" | "review" | "budget";
type Trials = {
  baseline: boolean;
  boundary: boolean;
  baseline_tokens: number | null;
  denied_calls: number | null;
};
type Lesson = {
  key: Key;
  title: string;
  role: string;
  topic: string;
  brief: string;
  goal: string;
  question: string;
  prompt: string;
  examples?: string[];
  greeting: string;
};
type Mode = "chat" | "document" | "answer";
type Turn = {
  budget_trials?: Trials;
  task: Key;
  request: string;
  text: string;
  reply: string;
  mode: Mode;
  tool: string;
  evidence: [string, string][];
  complete: boolean;
  at: string;
};
type Snapshot = {
  budget_trials: Trials;
  lessons: Lesson[];
  completed: Key[];
  conversations: Record<Key, Turn[]>;
};
const taskKeys = ["intro", "privacy", "release", "memory", "review", "budget"];
const video = "https://youtu.be/1WJdmvMoWgE";
export default function Room({
  session: defaultSession,
  model,
  exit,
}: {
  session: string;
  model?: React.ComponentProps<typeof Architecture>["model"];
  exit: () => void;
}) {
  const [session, setSession] = useState(
    () => localStorage.getItem("minigpt.room.session") || defaultSession,
  );
  const [data, load] = useState<Snapshot | null>(null),
    [key, select] = useState<Key>(() => {
      if (location.hash === "#llm10") return "budget";
      const k = localStorage.getItem("minigpt.room.task");
      return k && taskKeys.includes(k) ? (k as Key) : "intro";
    });
  const [architecture, showArchitecture] = useState(false);
  const [opened, open] = useState(location.hash === "#llm10"),
    [message, edit] = useState(""),
    [answer, setAnswer] = useState(""),
    [mode, setMode] = useState<"chat" | "document">("chat"),
    [working, wait] = useState(false),
    [error, fail] = useState(""),
    [revision, retry] = useState(0);
  const composer = useRef<HTMLTextAreaElement>(null),
    log = useRef<HTMLDivElement>(null),
    panel = useRef<HTMLElement>(null),
    pending = useRef(false),
    lastRequest = useRef({ signature: "", id: "" });
  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/room/" + session, { signal: controller.signal })
      .then(async (res) => {
        if (!res.ok) throw Error("Could not load this room.");
        load(await res.json());
        fail("");
      })
      .catch((e) => {
        if (e.name !== "AbortError") fail(e.message);
      });
    return () => controller.abort();
  }, [session, revision]);
  const lesson = data?.lessons.find((l) => l.key === key),
    turns = data?.conversations[key] || [],
    count = data?.completed.length || 0;
  useEffect(() => {
    if (log.current) log.current.scrollTop = log.current.scrollHeight;
  }, [key, opened, turns.length]);
  function switchTask(k: Key, show: boolean) {
    if (location.hash === "#llm10" && k !== "budget")
      window.history.replaceState(null, "", "#guided-room");
    select(k);
    localStorage.setItem("minigpt.room.task", k);
    open(show);
    setMode("chat");
    edit("");
    setAnswer("");
    fail("");
  }
  function openChat() {
    open(true);
    requestAnimationFrame(() => {
      panel.current?.scrollIntoView({ block: "nearest" });
      composer.current?.focus({ preventScroll: true });
    });
  }
  useEffect(() => {
    const followShortcut = () => {
      if (location.hash === "#llm10" && !pending.current) switchTask("budget", true);
    };
    window.addEventListener("hashchange", followShortcut);
    return () => window.removeEventListener("hashchange", followShortcut);
  }, []);
  async function send(text: string, kind: Mode) {
    if (!text.trim() || pending.current) return;
    const signature = JSON.stringify([key, kind, text]);
    if (signature !== lastRequest.current.signature)
      lastRequest.current = { signature, id: crypto.randomUUID() };
    pending.current = true;
    wait(true);
    fail("");
    open(true);
    try {
      const res = await fetch("/api/room/" + session + "/turn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task: key,
          text,
          mode: kind,
          request: lastRequest.current.id,
        }),
      });
      if (!res.ok)
        throw Error("The room could not process this turn. Try again.");
      const turn: Turn = await res.json();
      load((old) => {
        if (!old) return old;
        const history = old.conversations[turn.task];
        return {
          ...old,
          budget_trials: turn.budget_trials ?? old.budget_trials,
          completed:
            turn.complete && !old.completed.includes(turn.task)
              ? [...old.completed, turn.task]
              : old.completed,
          conversations: {
            ...old.conversations,
            [turn.task]: [
              ...history.filter((t) => t.request !== turn.request),
              turn,
            ].slice(-50),
          },
        };
      });
      if (turn.task !== key) switchTask(turn.task, true);
      if (kind === "answer") setAnswer("");
      else edit("");
      lastRequest.current = { signature: "", id: "" };
    } catch (e) {
      fail((e as Error).message);
    } finally {
      pending.current = false;
      wait(false);
    }
  }
  const idx = data?.lessons.findIndex((l) => l.key === key) ?? 0;
  const nextLesson =
    data &&
    [...data.lessons.slice(idx + 1), ...data.lessons.slice(0, idx)].find(
      (l) => !data.completed.includes(l.key),
    );
  return (
    <div className="room">
      <div className="room-top">
        <button
          onClick={exit}
          className="room-brand"
          aria-label="Open workspace"
        >
          <span>
            <Layers3 size={23} />
          </span>
          MiniGPT
        </button>
        <div className="room-crumb">
          <ChevronRight size={15} /> Learning rooms <ChevronRight size={15} />{" "}
          <strong>LLM security</strong>
        </div>
        <button onClick={exit} className="room-exit">
          <ArrowLeft size={15} /> Workspace
        </button>
      </div>
      <main className="room-main">
        <div className="room-intro">
          <div>
            <div className="room-kicker">
              <FlaskConical size={14} /> Hands-on room
            </div>
            <h1>LLM security field lab</h1>
            <p>
              Choose a task, try an example, and use the agent’s evidence to
              check your finding.
            </p>
          </div>
          <div className="room-links">
            <button
              className="room-video room-linkbtn"
              onClick={() => showArchitecture(true)}
            >
              <Workflow size={16} /> Agent architecture
            </button>
            <a
              className="room-video"
              href={video}
              target="_blank"
              rel="noreferrer"
            >
              <Video size={16} /> Companion video <ArrowRight size={13} />
            </a>
          </div>
        </div>
        {data && key !== "budget" && (
          <div className="room-feature">
            <div>
              <strong>OWASP LLM10 · Unbounded Consumption</strong>
              <p>Try one request at the limit and one just beyond it.</p>
            </div>
            <button
              disabled={working}
              onClick={() => switchTask("budget", true)}
            >
              Open exercise <ArrowRight size={15} />
            </button>
          </div>
        )}
        {architecture && (
          <Architecture
            last={turns.at(-1)}
            model={model}
            close={() => showArchitecture(false)}
          />
        )}
        {error && (
          <div className="room-error" role="alert">
            {error}{" "}
            {!data && (
              <button onClick={() => retry(revision + 1)}>Retry</button>
            )}
          </div>
        )}
        {!data || !lesson ? (
          <div role="status" className="room-loading">
            Loading room…
          </div>
        ) : (
          <div className="room-layout">
            <aside className="room-briefs" aria-label="Task briefs">
              <div className="room-rail-title">
                <BookOpen size={18} />
                <h2>Room tasks</h2>
                <strong>
                  {count} / {data.lessons.length}
                </strong>
              </div>
              <progress
                aria-label="Room progress"
                max={data.lessons.length}
                value={count}
              ></progress>
              {data.lessons.map((l, i) => {
                const done = data.completed.includes(l.key),
                  active = l.key === key;
                return (
                  <section
                    key={l.key}
                    className={"room-task" + (active ? " current" : "")}
                  >
                    <button
                      className="task-toggle"
                      disabled={working}
                      aria-expanded={active}
                      onClick={() => {
                        if (!active) switchTask(l.key, false);
                      }}
                    >
                      <span className={"task-num" + (done ? " done" : "")}>
                        {done ? (
                          <Check size={14} />
                        ) : (
                          String(i + 1).padStart(2, "0")
                        )}
                      </span>
                      <span>{l.title}</span>
                      {active ? (
                        <ChevronDown size={15} />
                      ) : (
                        <ChevronRight size={15} />
                      )}
                    </button>
                    {active && (
                      <div className="brief-body">
                        <span className="room-category">{l.topic}</span>
                        <p>{l.brief}</p>
                        <h3>Objective</h3>
                        <p>{l.goal}</p>
                        <button
                          className="launch"
                          disabled={working}
                          onClick={openChat}
                        >
                          <Bot size={16} />
                          {opened ? "Focus agent" : "Open agent"}
                          <ArrowRight size={14} />
                        </button>
                        <form
                          className="room-answer"
                          onSubmit={(e) => {
                            e.preventDefault();
                            send(answer, "answer");
                          }}
                        >
                          <label htmlFor="finding">{l.question}</label>
                          <div>
                            <input
                              id="finding"
                              value={answer}
                              maxLength={160}
                              onChange={(e) => setAnswer(e.target.value)}
                              placeholder="Your finding"
                              disabled={working || done}
                            />
                            <button
                              aria-label="Check answer"
                              title="Check answer"
                              disabled={working || done || !answer.trim()}
                            >
                              <Check size={18} />
                            </button>
                          </div>
                          {done && (
                            <small>
                              <CheckCheck size={14} /> Finding verified
                            </small>
                          )}
                          {turns.at(-1)?.tool === "answer.check" && !done && (
                            <span className="answer-feedback" role="status">
                              {turns.at(-1)?.reply}
                            </span>
                          )}
                        </form>
                        {done && nextLesson && (
                          <button
                            className="standby-button next-task"
                            disabled={working}
                            onClick={() => switchTask(nextLesson.key, true)}
                          >
                            Next task · {nextLesson.title}{" "}
                            <ArrowRight size={14} />
                          </button>
                        )}
                      </div>
                    )}
                  </section>
                );
              })}
              {count === data.lessons.length && (
                <div className="room-finished">
                  <GraduationCap /> All {data.lessons.length} findings verified{" "}
                  <button
                    className="standby-button"
                    disabled={working}
                    onClick={() => {
                      const next = crypto.randomUUID();
                      localStorage.setItem("minigpt.room.session", next);
                      load(null);
                      switchTask("intro", false);
                      setSession(next);
                    }}
                  >
                    New attempt
                  </button>
                </div>
              )}
            </aside>
            <section className="room-agent" ref={panel} aria-label="Task agent">
              <div className="agent-top">
                <div className="room-avatar">
                  <Bot size={22} />
                </div>
                <div>
                  <h2>{lesson.role}</h2>
                  <span>
                    Task {idx + 1} · {lesson.title}
                  </span>
                </div>
                <div className="agent-local">
                  <i />
                  Local
                </div>
              </div>
              {key === "budget" && lesson.examples && (
                <div className="budget-trials">
                  <div className="budget-caption">
                    <h3>Compare two requests</h3>
                    <a
                      target="_blank"
                      rel="noreferrer"
                      href="https://genai.owasp.org/llmrisk/llm102025-unbounded-consumption/"
                    >
                      OWASP LLM10 ↗
                    </a>
                  </div>
                  <div className="budget-grid">
                    {lesson.examples.map((sample, i) => (
                      <button
                        key={sample}
                        disabled={working}
                        onClick={() => send(sample, "chat")}
                      >
                        <strong>
                          {i === 0 ? "Run baseline" : "Test boundary"}{" "}
                          <ArrowRight size={13} />
                        </strong>
                        <code>{sample}</code>
                        {i === 0 && data.budget_trials.baseline && (
                          <small>
                            Allowed · {data.budget_trials.baseline_tokens}{" "}
                            tokens generated
                          </small>
                        )}
                        {i === 1 && data.budget_trials.boundary && (
                          <small>
                            Rejected · {data.budget_trials.denied_calls} model
                            calls
                          </small>
                        )}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <div
                className="room-log"
                ref={log}
                role="log"
                aria-label="Task conversation"
                aria-live="polite"
              >
                {!opened ? (
                  <div className="room-standby">
                    <div className="room-glyph">
                      <span />
                      <span />
                      <div>
                        <Bot size={34} />
                      </div>
                    </div>
                    <h2>Your task agent is ready</h2>
                    <p>Open the agent to work on “{lesson.title}”.</p>
                    <button className="standby-button" onClick={openChat}>
                      {turns.length
                        ? "Resume conversation"
                        : "Start conversation"}{" "}
                      <ArrowRight size={15} />
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="chat-day">
                      <span /> Task {idx + 1} briefing <span />
                    </div>
                    <div className="room-bubble guide-bubble">
                      <small>{lesson.role}</small>
                      <p>{lesson.greeting}</p>
                    </div>
                    {turns.map((t) => (
                      <div key={t.request} className="room-exchange">
                        <div className="room-bubble user-bubble">
                          <small>
                            {t.mode === "document"
                              ? "Document"
                              : t.mode === "answer"
                                ? "Your finding"
                                : "You"}
                          </small>
                          <p>{t.text}</p>
                        </div>
                        <div className="room-bubble guide-bubble">
                          <small>{lesson.role}</small>
                          <p>{t.reply}</p>
                          {t.evidence.length > 0 && (
                            <dl>
                              {t.evidence.map(([name, value]) => (
                                <div key={name}>
                                  <dt>{name}</dt>
                                  <dd>{value}</dd>
                                </div>
                              ))}
                            </dl>
                          )}
                          {t.tool !== "guide" && <code>{t.tool}</code>}
                        </div>
                      </div>
                    ))}
                    {working && (
                      <div role="status" className="room-thinking">
                        <LoaderCircle size={15} /> Checking…
                      </div>
                    )}
                  </>
                )}
              </div>
              <div className="room-compose">
                <div className="room-chips">
                  {key !== "budget" && (
                    <button
                      disabled={working}
                      onClick={() => send(lesson.prompt, "chat")}
                    >
                      {lesson.prompt}
                      <ArrowDownLeft size={12} />
                    </button>
                  )}
                  {key === "budget" && (
                    <button
                      disabled={
                        working ||
                        (!data.budget_trials.baseline &&
                          !data.budget_trials.boundary)
                      }
                      onClick={() => send("Explain the result", "chat")}
                    >
                      Explain result
                    </button>
                  )}
                  {key === "budget" &&
                    data.budget_trials.baseline &&
                    data.budget_trials.boundary &&
                    !data.completed.includes("budget") && (
                      <button
                        onClick={() =>
                          document.getElementById("finding")?.focus()
                        }
                      >
                        Enter finding <ArrowRight size={12} />
                      </button>
                    )}
                  <button
                    disabled={working}
                    onClick={() => send("Give me a hint", "chat")}
                  >
                    Hint
                  </button>
                  {key === "memory" && (
                    <button
                      disabled={working}
                      onClick={() => send("Read my preference", "chat")}
                    >
                      Read preference
                    </button>
                  )}
                </div>
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    send(message, mode);
                  }}
                >
                  <textarea
                    aria-label="Message room agent"
                    ref={composer}
                    value={message}
                    onFocus={() => open(true)}
                    onChange={(e) => edit(e.target.value)}
                    maxLength={1500}
                    rows={2}
                    placeholder={
                      mode === "document"
                        ? "Enter a document preference…"
                        : "Message the agent…"
                    }
                    onKeyDown={(e) => {
                      if (
                        e.key === "Enter" &&
                        !e.shiftKey &&
                        !e.nativeEvent.isComposing
                      ) {
                        e.preventDefault();
                        send(message, mode);
                      }
                    }}
                  />
                  <div className="compose-tools">
                    {key === "memory" ? (
                      <label>
                        Source{" "}
                        <select
                          aria-label="Room message source"
                          value={mode}
                          onChange={(e) =>
                            setMode(e.target.value as typeof mode)
                          }
                        >
                          <option value="chat">Operator</option>
                          <option value="document">Document</option>
                        </select>
                      </label>
                    ) : (
                      <span>
                        <LockKeyhole size={12} /> Local session
                      </span>
                    )}
                    <button
                      aria-label="Send room message"
                      disabled={working || !message.trim()}
                    >
                      <Send size={16} />
                    </button>
                  </div>
                </form>
              </div>
            </section>
          </div>
        )}
      </main>
    </div>
  );
}
