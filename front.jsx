import React, { useState, useRef, useEffect, useCallback } from "react";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
// Set VITE_API_URL in your .env file (frontend root) to override, e.g.:
//   VITE_API_URL=https://resolveai-hackathon.onrender.com
const API_URL = import.meta.env.VITE_API_URL || "https://resolveai-hackathon.onrender.com";

const TOKEN_KEY = "resolveai_token";
const EMAIL_KEY = "resolveai_email";

const STEPS = [
  "Understanding message",
  "Retrieving order data",
  "Reasoning about policy",
  "Taking action",
  "Logged",
];

let idCounter = 0;
const nextId = () => {
  idCounter += 1;
  return `${Date.now()}_${idCounter}`;
};

function messagesKey(email) {
  return `resolveai_messages_${email || "anon"}`;
}

function loadMessages(email) {
  try {
    const saved = localStorage.getItem(messagesKey(email));
    if (saved) {
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch (e) {
    console.error("Failed to load saved messages:", e);
  }
  return [
    {
      id: nextId(),
      role: "agent",
      decision: "answered",
      text: "Hi, I'm ResolveAI. Tell me what's going on with your order and I'll take it from there.",
      reasoning: null,
    },
  ];
}

async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(id);
  }
}

// ---------------------------------------------------------------------------
// Decision badge shown above certain agent messages
// ---------------------------------------------------------------------------
function DecisionBadge({ decision }) {
  if (!decision || decision === "answered") return null;

  const styles = {
    auto_resolve: { label: "RESOLVED", bg: "#DCFCE7", fg: "#166534" },
    escalate: { label: "ESCALATED", bg: "#FEF3C7", fg: "#92400E" },
    needs_info: { label: "NEEDS INFO", bg: "#DBEAFE", fg: "#1E40AF" },
  };
  const s = styles[decision];
  if (!s) return null;

  return (
    <span
      className="inline-block text-xs font-bold px-2 py-1 rounded-md mb-2"
      style={{ backgroundColor: s.bg, color: s.fg }}
    >
      {s.label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// A single chat bubble (agent or user), with optional reasoning expander
// ---------------------------------------------------------------------------
function MessageBubble({ msg, expanded, onToggle }) {
  const isUser = msg.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div className="bg-[#0F172A] text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-[75%] text-sm">
          {msg.text}
        </div>
      </div>
    );
  }

  const isAction = msg.decision === "auto_resolve" || msg.decision === "escalate";

  return (
    <div className="flex justify-start mb-4">
      <div
        className={
          "rounded-2xl rounded-tl-sm px-4 py-3 max-w-[75%] text-sm " +
          (isAction ? "" : "bg-[#F1F5F9] text-[#1E293B]")
        }
        style={
          isAction
            ? {
                backgroundColor: msg.decision === "auto_resolve" ? "#F0FDF4" : "#FFFBEB",
                border: `1px solid ${msg.decision === "auto_resolve" ? "#BBF7D0" : "#FDE68A"}`,
              }
            : undefined
        }
      >
        <DecisionBadge decision={msg.decision} />
        {isAction && (
          <div className="font-semibold text-[#1E293B] mb-0.5">{msg.text}</div>
        )}
        {!isAction && <div>{msg.text}</div>}

        {msg.reasoning && (
          <div className="mt-2">
            <button
              onClick={onToggle}
              className="text-xs text-[#1C7293] font-medium flex items-center gap-1"
            >
              <span>{expanded ? "▾" : "▸"}</span> Why did it decide this?
            </button>
            {expanded && (
              <div className="text-xs text-[#64748B] mt-1 border-l-2 border-[#CBD5E1] pl-2">
                {msg.reasoning}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Auth screen (login / signup)
// ---------------------------------------------------------------------------
function AuthScreen({ onAuthed }) {
  const [mode, setMode] = useState("login"); // "login" | "signup"
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [upiId, setUpiId] = useState("");
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    setInfo("");
    setLoading(true);
    try {
      if (mode === "signup") {
        const res = await fetch(`${API_URL}/signup`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password, upi_id: upiId }),
        });
        const data = await res.json();
        if (data.error) {
          setError(data.error);
        } else {
          setInfo(data.message || "Signup successful. You can now log in.");
          setMode("login");
        }
      } else {
        const res = await fetch(`${API_URL}/login`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.detail || "Invalid email or password.");
        }
        const data = await res.json();
        localStorage.setItem(TOKEN_KEY, data.access_token);
        localStorage.setItem(EMAIL_KEY, email);
        onAuthed(data.access_token, email);
      }
    } catch (err) {
      setError(err.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#F8FAFC] px-4">
      <div className="w-full max-w-sm bg-white rounded-2xl shadow-lg p-8">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-full bg-[#065A82] flex items-center justify-center text-white font-bold">
            R
          </div>
          <div>
            <div className="font-bold text-[#21295C] text-lg leading-tight">ResolveAI</div>
            <div className="text-xs text-[#64748B]">Autonomous support agent</div>
          </div>
        </div>

        <div className="flex mb-6 rounded-lg bg-[#F1F5F9] p-1">
          <button
            className={
              "flex-1 text-sm font-medium py-2 rounded-md transition " +
              (mode === "login" ? "bg-white shadow text-[#21295C]" : "text-[#64748B]")
            }
            onClick={() => { setMode("login"); setError(""); setInfo(""); }}
          >
            Log in
          </button>
          <button
            className={
              "flex-1 text-sm font-medium py-2 rounded-md transition " +
              (mode === "signup" ? "bg-white shadow text-[#21295C]" : "text-[#64748B]")
            }
            onClick={() => { setMode("signup"); setError(""); setInfo(""); }}
          >
            Sign up
          </button>
        </div>

        <form onSubmit={submit} className="space-y-3">
          <div>
            <label className="text-xs font-medium text-[#334155]">Email</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 w-full border border-[#CBD5E1] rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#1C7293]"
              placeholder="you@example.com"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-[#334155]">Password</label>
            <input
              type="password"
              required
              minLength={6}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 w-full border border-[#CBD5E1] rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#1C7293]"
              placeholder="••••••••"
            />
          </div>
          {mode === "signup" && (
            <div>
              <label className="text-xs font-medium text-[#334155]">
                UPI ID <span className="text-[#94A3B8]">(refunds are sent here)</span>
              </label>
              <input
                type="text"
                required
                value={upiId}
                onChange={(e) => setUpiId(e.target.value)}
                className="mt-1 w-full border border-[#CBD5E1] rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#1C7293]"
                placeholder="yourname@upi"
              />
            </div>
          )}

          {error && <div className="text-xs text-[#B91C1C] bg-[#FEF2F2] rounded-md px-3 py-2">{error}</div>}
          {info && <div className="text-xs text-[#166534] bg-[#F0FDF4] rounded-md px-3 py-2">{info}</div>}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-[#065A82] hover:bg-[#054A6B] transition text-white text-sm font-semibold rounded-lg py-2.5 disabled:opacity-60"
          >
            {loading ? "Please wait…" : mode === "login" ? "Log in" : "Create account"}
          </button>
        </form>

        {mode === "signup" && (
          <p className="text-[11px] text-[#94A3B8] mt-4 text-center">
            After signing up, check your email to confirm your account before logging in.
          </p>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main chat screen
// ---------------------------------------------------------------------------
function ChatScreen({ token, email, onLogout }) {
  const [messages, setMessages] = useState(() => loadMessages(email));
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [activeStep, setActiveStep] = useState(-1); // -1 = idle
  const [expanded, setExpanded] = useState({});
  const [auditFeed, setAuditFeed] = useState([]);
  const [wakingUp, setWakingUp] = useState(false);
  const [mobilePanelOpen, setMobilePanelOpen] = useState(false);

  const scrollRef = useRef(null);
  const stepTimerRef = useRef(null);
  const wakingTimerRef = useRef(null);

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
      localStorage.setItem(messagesKey(email), JSON.stringify(messages));
    } catch (e) {
      console.error("Failed to save messages:", e);
    }
  }, [messages, email]);

  const runStepAnimation = () => {
    setActiveStep(0);
    let step = 0;
    stepTimerRef.current = setInterval(() => {
      step += 1;
      if (step >= STEPS.length - 1) {
        clearInterval(stepTimerRef.current);
        setActiveStep(STEPS.length - 2); // hold just before "Logged" until response arrives
        return;
      }
      setActiveStep(step);
    }, 700);
  };

  const finishStepAnimation = () => {
    clearInterval(stepTimerRef.current);
    setActiveStep(STEPS.length - 1); // "Logged"
    setTimeout(() => setActiveStep(-1), 900);
  };

  const toggleReasoning = (id) => {
    setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const sendMessage = useCallback(
    async (rawText) => {
      const text = rawText.trim();
      if (!text || isLoading) return;

      setMessages((prev) => [...prev, { id: nextId(), role: "user", text }]);
      setInput("");
      setIsLoading(true);
      runStepAnimation();

      wakingTimerRef.current = setTimeout(() => setWakingUp(true), 4000);

      const attempt = async () => {
        const res = await fetchWithTimeout(
          `${API_URL}/chat`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({ customer_message: text }),
          },
          65000
        );
        if (res.status === 401) {
          onLogout();
          throw new Error("Session expired. Please log in again.");
        }
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

        if (data.error || !data.decision) {
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
          setAuditFeed((prev) => [...data.audit_log, ...prev].slice(0, 20));
        }
      } catch (err) {
        clearTimeout(wakingTimerRef.current);
        setWakingUp(false);
        clearInterval(stepTimerRef.current);
        setActiveStep(-1);
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: "agent",
            decision: null,
            text: "Something went wrong processing that message. Please try rephrasing or sending it again.",
            reasoning: null,
            isError: true,
          },
        ]);
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading, token, onLogout]
  );

  const quickReplies = [
    "My order #1234 never arrived",
    "My order #5678 arrived late",
    "What is your refund policy?",
    "hi",
  ];

  return (
    <div className="min-h-screen bg-white flex flex-col md:flex-row">
      {/* Header (mobile) */}
      <div className="md:hidden flex items-center justify-between px-4 py-3 border-b border-[#E2E8F0]">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-[#065A82] flex items-center justify-center text-white text-sm font-bold">R</div>
          <div>
            <div className="font-bold text-[#21295C] text-sm leading-tight">ResolveAI</div>
            <div className="text-[10px] text-[#64748B]">Autonomous support agent</div>
          </div>
        </div>
        <button onClick={() => setMobilePanelOpen((v) => !v)} className="text-xs text-[#1C7293] font-medium">
          Activity
        </button>
      </div>

      {/* Main chat column */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="hidden md:flex items-center justify-between px-6 py-4 border-b border-[#E2E8F0]">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-[#065A82] flex items-center justify-center text-white font-bold">R</div>
            <div>
              <div className="font-bold text-[#21295C] leading-tight">ResolveAI</div>
              <div className="text-xs text-[#64748B]">Autonomous support agent</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-[#64748B]">{email}</span>
            <button onClick={onLogout} className="text-xs font-medium text-[#B91C1C] hover:underline">
              Log out
            </button>
          </div>
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 md:px-6 py-6">
          {messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              msg={msg}
              expanded={!!expanded[msg.id]}
              onToggle={() => toggleReasoning(msg.id)}
            />
          ))}

          {isLoading && (
            <div className="flex justify-start mb-4">
              <div className="bg-[#F1F5F9] rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-[#64748B]">
                {wakingUp
                  ? "Still working on it — the server may be waking up from idle, this can take up to a minute…"
                  : "Thinking…"}
              </div>
            </div>
          )}
        </div>

        {/* Quick replies */}
        <div className="px-4 md:px-6 pb-2 flex flex-wrap gap-2">
          {quickReplies.map((q) => (
            <button
              key={q}
              onClick={() => sendMessage(q)}
              disabled={isLoading}
              className="text-xs border border-[#CBD5E1] text-[#334155] rounded-full px-3 py-1.5 hover:bg-[#F1F5F9] transition disabled:opacity-50"
            >
              {q}
            </button>
          ))}
        </div>

        {/* Input */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage(input);
          }}
          className="px-4 md:px-6 pb-4 md:pb-6 flex gap-2"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading}
            placeholder="Describe your issue..."
            className="flex-1 border border-[#CBD5E1] rounded-full px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-[#1C7293] disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="w-10 h-10 flex items-center justify-center rounded-full bg-[#065A82] text-white disabled:opacity-40 shrink-0"
            aria-label="Send"
          >
            ➤
          </button>
        </form>
      </div>

      {/* Agent activity side panel */}
      <div
        className={
          "w-full md:w-80 bg-[#0B1230] text-white flex-col " +
          (mobilePanelOpen ? "flex" : "hidden md:flex")
        }
      >
        <div className="px-5 py-5 border-b border-white/10">
          <div className="text-xs font-semibold tracking-wide text-[#93C5FD] flex items-center gap-2">
            <span>⚡</span> AGENT ACTIVITY
          </div>
        </div>

        <div className="px-5 py-5 space-y-4">
          {STEPS.map((label, i) => {
            const done = activeStep > i || activeStep === -1 && i === 0 ? false : activeStep > i;
            const current = activeStep === i;
            const isDone = activeStep === -1 ? false : activeStep > i;
            return (
              <div key={label} className="flex items-center gap-3">
                <div
                  className={
                    "w-3 h-3 rounded-full border-2 shrink-0 " +
                    (current
                      ? "bg-[#1C7293] border-[#1C7293] animate-pulse"
                      : isDone
                      ? "bg-[#1C7293] border-[#1C7293]"
                      : "border-white/30")
                  }
                />
                <span className={"text-sm " + (current ? "text-white font-medium" : "text-white/60")}>
                  {label}
                </span>
              </div>
            );
          })}
          <div className="text-[11px] text-white/40 pt-1">
            {activeStep === -1 ? "Idle — waiting for a message" : "Processing…"}
          </div>
        </div>

        <div className="px-5 py-5 border-t border-white/10 flex-1 overflow-y-auto">
          <div className="text-xs font-semibold tracking-wide text-[#93C5FD] mb-3">AUDIT LOG</div>
          {auditFeed.length === 0 ? (
            <div className="text-xs text-white/40">No actions logged yet.</div>
          ) : (
            <div className="space-y-3">
              {auditFeed.map((entry, i) => (
                <div key={i} className="text-xs">
                  <div className="font-mono text-[#5EEAD4] mb-0.5">
                    [{entry.order_id || "—"}] {entry.decision}
                  </div>
                  <div className="text-white/60">{entry.result}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Root component - decides auth vs chat
// ---------------------------------------------------------------------------
export default function ResolveAIChat() {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const [email, setEmail] = useState(() => localStorage.getItem(EMAIL_KEY));

  const handleAuthed = (newToken, newEmail) => {
    setToken(newToken);
    setEmail(newEmail);
  };

  const handleLogout = () => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(EMAIL_KEY);
    setToken(null);
    setEmail(null);
  };

  if (!token) {
    return <AuthScreen onAuthed={handleAuthed} />;
  }

  return <ChatScreen token={token} email={email} onLogout={handleLogout} />;
}
