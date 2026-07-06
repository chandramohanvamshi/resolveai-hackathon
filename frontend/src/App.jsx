import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Send,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  MessageCircle,
  ChevronDown,
  Loader2,
  Bot,
  User,
  Activity,
  Circle,
  X,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Update this when you deploy the FastAPI backend to a public URL.
// ---------------------------------------------------------------------------
const API_URL = "https://resolveai-hackathon.onrender.com/chat";

const STEPS = [
  "Understanding message",
  "Retrieving order data",
  "Reasoning about policy",
  "Taking action",
  "Logged",
];

const QUICK_MESSAGES = [
  "My order #1234 never arrived",
  "My order #5678 arrived late",
  "What is your refund policy?",
  "hi",
];

const DECISION_META = {
  auto_resolve: {
    label: "Resolved",
    icon: CheckCircle2,
    bubble: "bg-emerald-50 border border-emerald-200 text-emerald-900",
    iconColor: "text-emerald-600",
    tagColor: "bg-emerald-100 text-emerald-700",
    logColor: "text-emerald-400",
  },
  escalate: {
    label: "Escalated",
    icon: AlertTriangle,
    bubble: "bg-amber-50 border border-amber-200 text-amber-900",
    iconColor: "text-amber-600",
    tagColor: "bg-amber-100 text-amber-700",
    logColor: "text-amber-400",
  },
  needs_info: {
    label: "Needs info",
    icon: HelpCircle,
    bubble: "bg-sky-50 border border-sky-200 text-sky-900",
    iconColor: "text-sky-600",
    tagColor: "bg-sky-100 text-sky-700",
    logColor: "text-sky-400",
  },
  answered: {
    label: "Answered",
    icon: MessageCircle,
    bubble: "bg-slate-100 border border-slate-200 text-slate-800",
    iconColor: "text-slate-500",
    tagColor: "bg-slate-200 text-slate-700",
    logColor: "text-slate-400",
  },
};

let idCounter = 0;
const nextId = () => {
  idCounter += 1;
  return idCounter;
};

export default function ResolveAIChat() {
  const DEFAULT_GREETING = {
  id: nextId(),
  role: "agent",
  decision: "answered",
  text:
    "Hi, I'm ResolveAI. Tell me what's going on with your order and I'll take it from there.",
  reasoning: null,
};

const [messages, setMessages] = useState(() => {
  try {
    const saved = localStorage.getItem("resolveai_messages");
    if (saved) {
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed;
      }
    }
  } catch (e) {
    console.error("Failed to load saved messages:", e);
  }
  return [DEFAULT_GREETING];
});
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [wakingUp, setWakingUp] = useState(false);
  const [activeStep, setActiveStep] = useState(-1); // -1 = idle, STEPS.length = done
  const [auditFeed, setAuditFeed] = useState([]);
  const [expanded, setExpanded] = useState({});
  const [mobilePanelOpen, setMobilePanelOpen] = useState(false);

  const scrollRef = useRef(null);
  const stepTimerRef = useRef(null);
  const wakingTimerRef = useRef(null);

  const fetchWithTimeout = async (url, options, timeoutMs) => {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      return res;
    } finally {
      clearTimeout(id);
    }
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  useEffect(() => {
    return () => {
      clearInterval(stepTimerRef.current);
      clearTimeout(wakingTimerRef.current);
    };
  }, []);
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  useEffect(() => {
    return () => {
      clearInterval(stepTimerRef.current);
      clearTimeout(wakingTimerRef.current);
    };
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("resolveai_messages", JSON.stringify(messages));
    } catch (e) {
      console.error("Failed to save messages:", e);
    }
  }, [messages]);

  const toggleReasoning = (id) => {
    setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const runStepAnimation = () => {
    setActiveStep(0);
    let step = 0;
    stepTimerRef.current = setInterval(() => {
      step += 1;
      if (step >= STEPS.length - 1) {
        // hold on "Taking action" until the response actually arrives
        clearInterval(stepTimerRef.current);
        setActiveStep(STEPS.length - 2);
      } else {
        setActiveStep(step);
      }
    }, 500);
  };

  const finishStepAnimation = () => {
    clearInterval(stepTimerRef.current);
    setActiveStep(STEPS.length - 1); // "Logged"
    setTimeout(() => setActiveStep(-1), 1200);
  };

  const sendMessage = useCallback(
    async (rawText) => {
      const text = rawText.trim();
      if (!text || isLoading) return;

      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: "user", text },
      ]);
      setInput("");
      setIsLoading(true);
      runStepAnimation();

      // If the first attempt takes more than 4s, it's likely a cold start on the free
      // hosting tier - let the person know rather than leaving them guessing.
      wakingTimerRef.current = setTimeout(() => setWakingUp(true), 4000);

      const attempt = async () => {
        const res = await fetchWithTimeout(
          API_URL,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ customer_message: text }),
          },
          65000
        );
        if (!res.ok) throw new Error(`Server responded with ${res.status}`);
        return res.json();
      };

      try {
        let data;
        try {
          data = await attempt();
        } catch (firstErr) {
          // One silent retry - covers the case where the free-tier server was
          // asleep and the first request woke it up but didn't complete in time.
          data = await attempt();
        }

        // The backend returns { error, detail } instead of a normal decision
        // payload when something goes wrong server-side - treat that as a
        // failure rather than rendering a blank message bubble.
        if (data.error || !data.decision) {
          console.error("Backend error response:", data);
          throw new Error(data.detail || data.error || "Empty response from server");
        }

        clearTimeout(wakingTimerRef.current);
        setWakingUp(false);
        finishStepAnimation();

        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: "agent",
            decision: data.decision,
            text: data.action_result,
            reasoning: data.reasoning,
          },
        ]);

        if (Array.isArray(data.audit_log) && data.audit_log.length > 0) {
          setAuditFeed((prev) => [...data.audit_log.slice().reverse(), ...prev].slice(0, 20));
        }
      } catch (err) {
        clearTimeout(wakingTimerRef.current);
        setWakingUp(false);
        finishStepAnimation();
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: "error",
            text:
              "Something went wrong processing that message. Please try rephrasing or sending it again.",
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading]
  );

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  return (
    <div className="h-screen w-full bg-slate-100 flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-4 sm:px-6 py-3 bg-white border-b border-slate-200 shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-lg bg-slate-900 flex items-center justify-center">
            <Bot className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-slate-900 leading-tight">
              ResolveAI
            </h1>
            <p className="text-xs text-slate-500 leading-tight">
              Autonomous support agent
            </p>
          </div>
        </div>
        <button
          onClick={() => setMobilePanelOpen(true)}
          className="lg:hidden flex items-center gap-1.5 text-xs font-medium text-slate-600 bg-slate-100 border border-slate-200 rounded-full px-3 py-1.5"
        >
          <Activity className="w-3.5 h-3.5 text-cyan-500" />
          Agent activity
          {isLoading && (
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse" />
          )}
        </button>
      </header>

      <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[1fr_320px]">
        {/* Chat column */}
        <div className="flex flex-col min-h-0 bg-slate-50">
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 sm:px-6 py-4 space-y-4">
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                msg={msg}
                expanded={!!expanded[msg.id]}
                onToggle={() => toggleReasoning(msg.id)}
              />
            ))}

            {isLoading && (
              <div className="flex items-start gap-2.5">
                <div className="w-7 h-7 rounded-full bg-slate-900 flex items-center justify-center shrink-0">
                  <Bot className="w-4 h-4 text-cyan-400" />
                </div>
                <div>
                  <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-4 py-2.5 flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:-0.3s]" />
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:-0.15s]" />
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce" />
                  </div>
                  {wakingUp && (
                    <p className="text-[11px] text-slate-500 mt-1.5 px-1 max-w-[220px]">
                      Still working — the server may be waking up from idle. This can take up to a minute.
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Quick test buttons */}
          <div className="px-3 sm:px-6 pt-2 pb-1 shrink-0">
            <div className="flex flex-wrap gap-2">
              {QUICK_MESSAGES.map((q) => (
                <button
                  key={q}
                  onClick={() => sendMessage(q)}
                  disabled={isLoading}
                  className="text-xs px-3 py-1.5 rounded-full border border-slate-300 bg-white text-slate-600 hover:border-cyan-400 hover:text-cyan-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>

          {/* Input */}
          <div className="px-3 sm:px-6 py-3 bg-white border-t border-slate-200 shrink-0">
            <div className="flex items-end gap-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Describe your issue..."
                rows={1}
                className="flex-1 resize-none rounded-xl border border-slate-300 px-4 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent max-h-28"
              />
              <button
                onClick={() => sendMessage(input)}
                disabled={isLoading || !input.trim()}
                className="w-10 h-10 shrink-0 rounded-xl bg-slate-900 text-white flex items-center justify-center hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                aria-label="Send message"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Desktop activity panel */}
        <div className="hidden lg:block border-l border-slate-800">
          <ActivityPanel activeStep={activeStep} auditFeed={auditFeed} isLoading={isLoading} />
        </div>
      </div>

      {/* Mobile activity panel drawer */}
      {mobilePanelOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div
            className="absolute inset-0 bg-black/40"
            onClick={() => setMobilePanelOpen(false)}
          />
          <div className="absolute right-0 top-0 bottom-0 w-[85%] max-w-sm">
            <div className="relative h-full">
              <button
                onClick={() => setMobilePanelOpen(false)}
                className="absolute top-3 right-3 z-10 w-7 h-7 rounded-full bg-slate-800 text-slate-300 flex items-center justify-center"
              >
                <X className="w-4 h-4" />
              </button>
              <ActivityPanel activeStep={activeStep} auditFeed={auditFeed} isLoading={isLoading} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ActivityPanel({ activeStep, auditFeed, isLoading }) {
  return (
    <div className="h-full bg-slate-900 text-slate-300 flex flex-col font-mono">
      <div className="px-4 py-3 border-b border-slate-800 flex items-center gap-2 shrink-0">
        <Activity className="w-4 h-4 text-cyan-400" />
        <span className="text-xs uppercase tracking-wider text-slate-400">
          Agent activity
        </span>
        <span
          className={`ml-auto w-2 h-2 rounded-full ${
            isLoading ? "bg-cyan-400 animate-pulse" : "bg-slate-700"
          }`}
        />
      </div>

      {/* Step tracker */}
      <div className="px-4 py-5 border-b border-slate-800 shrink-0">
        <ol className="space-y-0">
          {STEPS.map((step, i) => {
            const isDone = activeStep > i || activeStep === -1;
            const isActive = activeStep === i;
            const isPending = activeStep < i && activeStep !== -1;
            const isLast = i === STEPS.length - 1;
            return (
              <li key={step} className="relative flex items-start gap-3 pb-5 last:pb-0">
                {!isLast && (
                  <span
                    className={`absolute left-[9px] top-5 w-px h-full ${
                      isDone && activeStep !== -1 ? "bg-cyan-500" : "bg-slate-700"
                    }`}
                  />
                )}
                <span className="relative shrink-0 mt-0.5">
                  {isActive ? (
                    <Loader2 className="w-[18px] h-[18px] text-cyan-400 animate-spin" />
                  ) : isDone && activeStep !== -1 ? (
                    <CheckCircle2 className="w-[18px] h-[18px] text-cyan-500" />
                  ) : activeStep === -1 ? (
                    <Circle className="w-[18px] h-[18px] text-slate-700" />
                  ) : (
                    <Circle className="w-[18px] h-[18px] text-slate-700" />
                  )}
                </span>
                <span
                  className={`text-xs pt-0.5 ${
                    isActive
                      ? "text-cyan-300"
                      : isPending
                      ? "text-slate-600"
                      : "text-slate-400"
                  }`}
                >
                  {step}
                </span>
              </li>
            );
          })}
        </ol>
        {activeStep === -1 && !isLoading && (
          <p className="text-[11px] text-slate-600 mt-1">Idle — waiting for a message</p>
        )}
      </div>

      {/* Audit log feed */}
      <div className="flex-1 min-h-0 flex flex-col">
        <div className="px-4 pt-3 pb-2 text-[11px] uppercase tracking-wider text-slate-500 shrink-0">
          Audit log
        </div>
        <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-2">
          {auditFeed.length === 0 && (
            <p className="text-[11px] text-slate-600">No actions logged yet.</p>
          )}
          {auditFeed.map((entry, i) => {
            const meta = DECISION_META[entry.decision] || DECISION_META.answered;
            return (
              <div
                key={i}
                className="text-[11px] leading-relaxed border-l-2 border-slate-700 pl-2.5"
              >
                <div className="flex items-center gap-1.5">
                  <span className={meta.logColor}>[{entry.order_id || "n/a"}]</span>
                  <span className={meta.logColor}>{entry.decision}</span>
                </div>
                <p className="text-slate-500 truncate">{entry.result}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ msg, expanded, onToggle }) {
  if (msg.role === "user") {
    return (
      <div className="flex items-start gap-2.5 justify-end">
        <div className="bg-slate-900 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 max-w-[80%] sm:max-w-[65%] text-sm">
          {msg.text}
        </div>
        <div className="w-7 h-7 rounded-full bg-slate-200 flex items-center justify-center shrink-0">
          <User className="w-4 h-4 text-slate-500" />
        </div>
      </div>
    );
  }

  if (msg.role === "error") {
    return (
      <div className="flex items-start gap-2.5">
        <div className="w-7 h-7 rounded-full bg-slate-900 flex items-center justify-center shrink-0">
          <Bot className="w-4 h-4 text-cyan-400" />
        </div>
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-2xl rounded-tl-sm px-4 py-2.5 max-w-[80%] sm:max-w-[65%] text-sm">
          {msg.text}
        </div>
      </div>
    );
  }

  const meta = DECISION_META[msg.decision] || DECISION_META.answered;
  const Icon = meta.icon;
  const isPlain = msg.decision === "answered" || msg.decision === "needs_info";

  return (
    <div className="flex items-start gap-2.5">
      <div className="w-7 h-7 rounded-full bg-slate-900 flex items-center justify-center shrink-0">
        <Bot className="w-4 h-4 text-cyan-400" />
      </div>
      <div className="max-w-[85%] sm:max-w-[70%]">
        <div className={`rounded-2xl rounded-tl-sm px-4 py-2.5 text-sm ${meta.bubble}`}>
          {!isPlain && (
            <div className="flex items-center gap-1.5 mb-1">
              <Icon className={`w-3.5 h-3.5 ${meta.iconColor}`} />
              <span className={`text-[10px] font-semibold uppercase tracking-wide px-1.5 py-0.5 rounded-full ${meta.tagColor}`}>
                {meta.label}
              </span>
            </div>
          )}
          <p>{msg.text}</p>
        </div>

        {msg.reasoning && (
          <div className="mt-1">
            <button
              onClick={onToggle}
              className="flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-700 px-1"
            >
              <ChevronDown
                className={`w-3 h-3 transition-transform ${expanded ? "rotate-180" : ""}`}
              />
              Why did it decide this?
            </button>
            {expanded && (
              <div className="mt-1.5 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-[11px] text-slate-600 leading-relaxed">
                {msg.reasoning}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
