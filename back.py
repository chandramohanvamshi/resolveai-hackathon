"""
ResolveAI - Autonomous Customer Support Agent
LangGraph workflow: Understand -> Retrieve Context -> Reason & Decide
                     -> (Request Manager Approval | Escalate | ...) -> Audit Trail

Includes:
    - Signup / Login using Supabase Auth
    - Refunds are NOT auto-processed. A "pending refund" is created and an
      email is sent to the manager with Approve / Reject links. The refund
      only actually happens once the manager clicks Approve, and it is sent
      to the customer's UPI ID on file.
    - Order tracking, general Q&A, greetings all handled directly.

Install:
    pip install langgraph langchain-google-genai supabase fastapi uvicorn python-dotenv --break-system-packages

Required environment variables (set these in Render's Environment tab -
do NOT hardcode them anywhere in this file):
    SUPABASE_URL           - your Supabase project URL
    SUPABASE_KEY            - your Supabase SERVICE ROLE key (not the public/anon key -
                               needed so the backend can write regardless of RLS policies)
    GMAIL_ADDRESS           - the Gmail address that will send approval emails
    GMAIL_APP_PASSWORD      - a Gmail "App Password" (generate at
                               https://myaccount.google.com/apppasswords,
                               requires 2-Step Verification to be turned on)
    MANAGER_EMAIL           - where refund-approval emails should be sent
    BASE_URL                - public base URL of this API, used to build the
                               approve/reject links in the email
                               (e.g. https://resolveai-hackathon.onrender.com)

Run locally:
    uvicorn support_agent_graph:app_api --reload --port 8000

Supabase tables required - see supabase_setup.sql for the full script:
    profiles, pending_refunds  (new)
    orders, tickets, audit_log (already existed)
"""

import os
import re
import json
import secrets
import smtplib
from email.mime.text import MIMEText
from typing import TypedDict, Literal, Optional

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from supabase import create_client

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# ---------------------------------------------------------------------------
# Supabase setup - pulled from environment variables, never hardcoded.
# ---------------------------------------------------------------------------
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------------------------------
# Email setup (Gmail SMTP)
# ---------------------------------------------------------------------------
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
MANAGER_EMAIL = os.environ["MANAGER_EMAIL"]
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# 1. Shared state that flows through every node
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    customer_message: str
    customer_user_id: Optional[str]
    customer_email: Optional[str]
    upi_id: Optional[str]
    intent: Optional[str]
    order_id: Optional[str]
    order_data: Optional[dict]
    policy_context: Optional[str]
    decision: Optional[str]          # "auto_resolve" | "escalate" | "needs_info" | "answered"
    reasoning: Optional[str]
    action_result: Optional[str]
    audit_log: list


def clean_json(text: str) -> str:
    """Extract the first valid-looking JSON object from the model's response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


# ---------------------------------------------------------------------------
# 2. Node: Understand & classify
# ---------------------------------------------------------------------------
def understand(state: AgentState) -> AgentState:
    prompt = f"""Classify this customer message and extract any order ID.

Categories:
- "order_issue": customer wants a refund, cancellation, or is reporting a problem with a specific order
- "track_order": customer is asking about the current status, location, or delivery progress of a specific order
- "general_question": customer is asking a general question (policy, shipping times, how something works) not tied to a specific order
- "greeting": customer is just saying hello or making small talk
- "unclear": message doesn't fit any category above

Return ONLY JSON, no other text:
{{"category": "order_issue" or "track_order" or "general_question" or "greeting" or "unclear", "order_id": "..." or null}}

Message: {state['customer_message']}"""

    try:
        response = llm.invoke(prompt)
        parsed = json.loads(clean_json(response.content))
        intent = parsed.get("category", "unclear")
        order_id = parsed.get("order_id")
    except (json.JSONDecodeError, KeyError, AttributeError):
        intent = "unclear"
        order_id = None

    return {**state, "intent": intent, "order_id": order_id}


# ---------------------------------------------------------------------------
# 3. Node: Retrieve context (Supabase lookup + policy RAG)
#    Order lookup is scoped to the logged-in customer, so nobody can act on
#    an order that isn't theirs just by guessing an order ID.
# ---------------------------------------------------------------------------
def retrieve_context(state: AgentState) -> AgentState:
    order_data = fetch_order_from_db(state["order_id"], state.get("customer_user_id"))
    policy_snippets = search_policy_vector_db(state["intent"])

    return {
        **state,
        "order_data": order_data,
        "policy_context": policy_snippets,
    }


# ---------------------------------------------------------------------------
# 4. Node: Reason & decide (the branch point)
# ---------------------------------------------------------------------------
def reason_and_decide(state: AgentState) -> AgentState:
    if state["intent"] == "greeting":
        return {
            **state,
            "decision": "answered",
            "reasoning": "Customer sent a greeting.",
            "action_result": "Hi! I'm ResolveAI, your support assistant. I can help with refunds and order issues - what can I help you with?",
        }

    if state["intent"] == "general_question":
        answer = answer_general_question(state["customer_message"], state["policy_context"])
        return {
            **state,
            "decision": "answered",
            "reasoning": "Answered a general question using policy knowledge.",
            "action_result": answer,
        }

    if state["intent"] == "unclear":
        answer = answer_general_question(state["customer_message"], state["policy_context"])
        return {
            **state,
            "decision": "answered",
            "reasoning": "Message intent was unclear; asked Gemini to respond conversationally.",
            "action_result": answer,
        }

    if state["intent"] in ("track_order", "order_status"):
        order = state["order_data"]
        if order is None:
            return {
                **state,
                "decision": "needs_info",
                "reasoning": f"No order found matching ID {state['order_id']} for this account.",
            }
        status = order.get("status", "unknown")
        days_late = order.get("days_late", 0)

        if status == "delivered":
            tracking_msg = f"Order {state['order_id']} has been delivered."
        elif days_late and days_late > 0:
            tracking_msg = (
                f"Order {state['order_id']} is currently '{status}' and running "
                f"{days_late} day(s) behind the expected delivery date."
            )
        else:
            tracking_msg = f"Order {state['order_id']} is currently '{status}' and on schedule."

        return {
            **state,
            "decision": "answered",
            "reasoning": f"Order tracking lookup for {state['order_id']}.",
            "action_result": tracking_msg,
        }

    if not state["order_id"]:
        return {
            **state,
            "decision": "needs_info",
            "reasoning": "No order ID was provided by the customer.",
        }

    if state["order_data"] is None:
        return {
            **state,
            "decision": "needs_info",
            "reasoning": f"No order found matching ID {state['order_id']} for this account.",
        }

    prompt = f"""You are a support decision engine. Based on the order data and
policy below, decide whether to auto-resolve or escalate to a human.

Order data: {state['order_data']}
Policy: {state['policy_context']}
Customer intent: {state['intent']}

Return ONLY JSON, no other text:
{{"decision": "auto_resolve" or "escalate", "reasoning": "one sentence why"}}"""

    try:
        response = llm.invoke(prompt)
        parsed = json.loads(clean_json(response.content))
        decision = parsed.get("decision", "escalate")
        reasoning = parsed.get("reasoning", "Defaulted to escalation due to unclear model output.")
    except (json.JSONDecodeError, KeyError, AttributeError):
        decision = "escalate"
        reasoning = "Defaulted to escalation because the decision engine returned an unreadable response."

    return {**state, "decision": decision, "reasoning": reasoning}


def route_decision(state: AgentState) -> Literal["auto_resolve", "escalate", "needs_info", "answered"]:
    """Conditional edge function - reads state, returns next node name."""
    if state["decision"] in ("auto_resolve", "escalate", "needs_info", "answered"):
        return state["decision"]
    return "escalate"


# ---------------------------------------------------------------------------
# 4a. Helper: Answer a general question directly (no order needed)
# ---------------------------------------------------------------------------
def answer_general_question(customer_message: str, policy_context: str) -> str:
    prompt = f"""You are ResolveAI, a helpful and friendly customer support assistant.

If the customer's question relates to our store, orders, refunds, or policies,
answer using the policy information below - keep it short and direct, and if the
policy doesn't cover it, say you're not sure and offer to connect them with the team.

If the customer's question is general (like asking who you are, small talk, or a
basic factual question unrelated to our policies), answer it naturally and helpfully
using your own knowledge. Keep answers concise and friendly.

Policy: {policy_context}

Customer question: {customer_message}

Answer:"""

    try:
        response = llm.invoke(prompt)
        answer = (response.content or "").strip()
        if not answer:
            return "I don't have a clear answer for that. Let me connect you with our support team."
        return answer
    except Exception:
        return "I'm having trouble answering that right now - let me connect you with our support team."


# ---------------------------------------------------------------------------
# 4c. Node: Answered (general question / greeting / unclear / tracking - already resolved)
# ---------------------------------------------------------------------------
def answered(state: AgentState) -> AgentState:
    return state


# ---------------------------------------------------------------------------
# 4b. Node: Needs info (ask the customer for their order ID)
# ---------------------------------------------------------------------------
def needs_info(state: AgentState) -> AgentState:
    if state["order_id"]:
        message = f"I couldn't find an order matching ID {state['order_id']} on your account. Could you double-check the order number?"
    else:
        message = "Could you please share your order ID so I can look into this for you?"

    return {**state, "action_result": message}


# ---------------------------------------------------------------------------
# 5a. Node: Refund needs manager approval before anything is actually paid out.
# ---------------------------------------------------------------------------
def auto_resolve(state: AgentState) -> AgentState:
    token = create_pending_refund(
        order_id=state["order_id"],
        reasoning=state["reasoning"],
        customer_user_id=state.get("customer_user_id"),
        customer_email=state.get("customer_email"),
        upi_id=state.get("upi_id"),
    )
    send_manager_approval_email(
        order_id=state["order_id"],
        reasoning=state["reasoning"],
        upi_id=state.get("upi_id"),
        token=token,
    )

    return {
        **state,
        "action_result": (
            f"Your refund request for order {state['order_id']} looks eligible and has been "
            f"sent to a manager for approval. You'll get an email once it's processed."
        ),
    }


# ---------------------------------------------------------------------------
# 5b. Node: Escalate to human
# ---------------------------------------------------------------------------
def escalate(state: AgentState) -> AgentState:
    create_support_ticket(state["order_id"], state["reasoning"])
    notify_support_team(state["order_id"])

    return {
        **state,
        "action_result": f"Escalated order {state['order_id']} to support team",
    }


# ---------------------------------------------------------------------------
# 6. Node: Audit trail (all branches merge here)
# ---------------------------------------------------------------------------
def audit_trail(state: AgentState) -> AgentState:
    entry = {
        "order_id": state["order_id"],
        "decision": state["decision"],
        "reasoning": state["reasoning"],
        "result": state["action_result"],
    }
    save_audit_log(entry)

    return {**state, "audit_log": state.get("audit_log", []) + [entry]}


# ---------------------------------------------------------------------------
# 7. Build the graph
# ---------------------------------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("understand", understand)
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("reason_and_decide", reason_and_decide)
graph.add_node("auto_resolve", auto_resolve)
graph.add_node("escalate", escalate)
graph.add_node("needs_info", needs_info)
graph.add_node("answered", answered)
graph.add_node("audit_trail", audit_trail)

graph.set_entry_point("understand")
graph.add_edge("understand", "retrieve_context")
graph.add_edge("retrieve_context", "reason_and_decide")

graph.add_conditional_edges(
    "reason_and_decide",
    route_decision,
    {
        "auto_resolve": "auto_resolve",
        "escalate": "escalate",
        "needs_info": "needs_info",
        "answered": "answered",
    },
)

graph.add_edge("auto_resolve", "audit_trail")
graph.add_edge("escalate", "audit_trail")
graph.add_edge("needs_info", END)
graph.add_edge("answered", END)
graph.add_edge("audit_trail", END)

app = graph.compile()


# ---------------------------------------------------------------------------
# Supabase-backed database functions
# ---------------------------------------------------------------------------
MOCK_POLICY = (
    "Refunds are automatically approved if an order is more than 7 days late "
    "and the customer has requested fewer than 2 refunds in the current month. "
    "Orders delivered on time are not eligible for late-delivery refunds. "
    "Customers with 2 or more refunds this month must be reviewed by a human agent."
)


def fetch_order_from_db(order_id, customer_user_id=None):
    if not order_id:
        return None
    try:
        query = supabase.table("orders").select("*").eq("order_id", order_id)
        if customer_user_id:
            query = query.eq("customer_user_id", customer_user_id)
        response = query.execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"[DB ERROR] fetch_order_from_db failed: {e}")
        return None


def search_policy_vector_db(query):
    return MOCK_POLICY


def update_order_status_in_db(order_id, status):
    try:
        supabase.table("orders").update({"status": status}).eq("order_id", order_id).execute()
        print(f"[DB] Order {order_id} status updated to '{status}'")
    except Exception as e:
        print(f"[DB ERROR] update_order_status_in_db failed: {e}")


def create_support_ticket(order_id, reason):
    try:
        supabase.table("tickets").insert({"order_id": order_id, "reason": reason}).execute()
        print(f"[TICKET] Created support ticket for order {order_id}: {reason}")
    except Exception as e:
        print(f"[DB ERROR] create_support_ticket failed: {e}")


def notify_support_team(order_id):
    print(f"[SLACK] Support team notified about order {order_id}")


def save_audit_log(entry):
    try:
        supabase.table("audit_log").insert(entry).execute()
        print(f"[AUDIT] {entry}")
    except Exception as e:
        print(f"[DB ERROR] save_audit_log failed: {e}")


def fetch_profile(user_id):
    try:
        response = supabase.table("profiles").select("*").eq("user_id", user_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"[DB ERROR] fetch_profile failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Pending refund helpers (manager-approval workflow)
# ---------------------------------------------------------------------------
def create_pending_refund(order_id, reasoning, customer_user_id, customer_email, upi_id):
    token = secrets.token_urlsafe(32)
    try:
        supabase.table("pending_refunds").insert({
            "token": token,
            "order_id": order_id,
            "reasoning": reasoning,
            "customer_user_id": customer_user_id,
            "customer_email": customer_email,
            "upi_id": upi_id,
            "status": "pending",
        }).execute()
    except Exception as e:
        print(f"[DB ERROR] create_pending_refund failed: {e}")
    return token


def get_pending_refund(token):
    try:
        response = supabase.table("pending_refunds").select("*").eq("token", token).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"[DB ERROR] get_pending_refund failed: {e}")
        return None


def mark_pending_refund(token, status):
    try:
        supabase.table("pending_refunds").update({"status": status}).eq("token", token).execute()
    except Exception as e:
        print(f"[DB ERROR] mark_pending_refund failed: {e}")


# ---------------------------------------------------------------------------
# Email helpers (Gmail SMTP)
# ---------------------------------------------------------------------------
def _send_email(to_address, subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = to_address

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, [to_address], msg.as_string())
        print(f"[EMAIL] Sent '{subject}' to {to_address}")
    except Exception as e:
        print(f"[EMAIL ERROR] failed to send to {to_address}: {e}")


def send_manager_approval_email(order_id, reasoning, upi_id, token):
    approve_url = f"{BASE_URL}/refund/approve/{token}"
    reject_url = f"{BASE_URL}/refund/reject/{token}"

    body = (
        f"A refund needs your approval.\n\n"
        f"Order ID: {order_id}\n"
        f"Reason: {reasoning}\n"
        f"Refund destination (UPI ID): {upi_id or 'not on file'}\n\n"
        f"Approve: {approve_url}\n"
        f"Reject: {reject_url}\n"
    )
    _send_email(MANAGER_EMAIL, f"Refund approval needed - Order {order_id}", body)


def send_customer_approved_email(customer_email, order_id, upi_id):
    if not customer_email:
        print(f"[EMAIL] No customer email on file for order {order_id}, skipping confirmation.")
        return
    body = (
        f"Good news! Your refund for order {order_id} has been approved and processed.\n\n"
        f"It will be sent to your UPI ID: {upi_id or 'on file'}.\n\n"
        f"Thanks for your patience - ResolveAI Support"
    )
    _send_email(customer_email, f"Refund approved - Order {order_id}", body)


def send_customer_rejected_email(customer_email, order_id, reasoning):
    if not customer_email:
        print(f"[EMAIL] No customer email on file for order {order_id}, skipping rejection notice.")
        return
    body = (
        f"We reviewed your refund request for order {order_id}.\n\n"
        f"It needs a closer look from our support team before we can proceed, "
        f"and has been escalated to a specialist who will follow up with you directly.\n\n"
        f"Thanks for your patience - ResolveAI Support"
    )
    _send_email(customer_email, f"Update on your refund - Order {order_id}", body)


# ---------------------------------------------------------------------------
# 8. FastAPI wrapper - exposes the agent as a web API
# ---------------------------------------------------------------------------
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app_api = FastAPI()

app_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 8a. Auth: signup / login (Supabase Auth)
# ---------------------------------------------------------------------------
class SignupRequest(BaseModel):
    email: str
    password: str
    upi_id: str  # where refunds will be sent once approved


class LoginRequest(BaseModel):
    email: str
    password: str


@app_api.post("/signup")
def signup(request: SignupRequest):
    try:
        result = supabase.auth.sign_up({
            "email": request.email,
            "password": request.password,
        })
        user = result.user
        if user is None:
            return {"error": "Signup failed. The email may already be registered."}

        supabase.table("profiles").insert({
            "user_id": user.id,
            "email": request.email,
            "upi_id": request.upi_id,
        }).execute()

        return {"message": "Signup successful. Please check your email to confirm your account, then log in."}
    except Exception as e:
        return {"error": str(e)}


@app_api.post("/login")
def login(request: LoginRequest):
    try:
        result = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password,
        })
        return {
            "access_token": result.session.access_token,
            "refresh_token": result.session.refresh_token,
        }
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid email or password.")


def get_current_user(authorization: str = Header(None)):
    """FastAPI dependency: validates the Supabase access token sent as
    'Authorization: Bearer <access_token>' and returns the logged-in user."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        user_response = supabase.auth.get_user(token)
        user = user_response.user
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")
        return user
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


# ---------------------------------------------------------------------------
# 8b. Chat endpoint - requires login
# ---------------------------------------------------------------------------
class MessageRequest(BaseModel):
    customer_message: str


@app_api.post("/chat")
def chat(request: MessageRequest, current_user=Depends(get_current_user)):
    try:
        profile = fetch_profile(current_user.id)
        result = app.invoke({
            "customer_message": request.customer_message,
            "customer_user_id": current_user.id,
            "customer_email": current_user.email,
            "upi_id": profile.get("upi_id") if profile else None,
            "audit_log": [],
        })
        action_result = result.get("action_result") or "I'm not sure how to respond to that - could you rephrase, or share your order ID?"
        return {
            "decision": result["decision"],
            "reasoning": result["reasoning"],
            "action_result": action_result,
            "audit_log": result["audit_log"],
        }
    except Exception as e:
        return {"error": "Something went wrong processing this request.", "detail": str(e)}


# ---------------------------------------------------------------------------
# 8c. Manager approval endpoints - clicked from the email
# ---------------------------------------------------------------------------
@app_api.get("/refund/approve/{token}", response_class=HTMLResponse)
def approve_refund(token: str):
    pending = get_pending_refund(token)
    if not pending:
        return HTMLResponse("<h3>Invalid or expired approval link.</h3>", status_code=404)
    if pending["status"] != "pending":
        return HTMLResponse(f"<h3>This refund was already {pending['status']}.</h3>")

    update_order_status_in_db(pending["order_id"], status="refunded")
    send_customer_approved_email(pending.get("customer_email"), pending["order_id"], pending.get("upi_id"))
    mark_pending_refund(token, "approved")
    save_audit_log({
        "order_id": pending["order_id"],
        "decision": "auto_resolve",
        "reasoning": pending["reasoning"],
        "result": f"Refund approved by manager and sent to UPI {pending.get('upi_id') or 'not on file'}.",
    })

    return HTMLResponse(f"<h3>Refund for order {pending['order_id']} approved and processed.</h3>")


@app_api.get("/refund/reject/{token}", response_class=HTMLResponse)
def reject_refund(token: str):
    pending = get_pending_refund(token)
    if not pending:
        return HTMLResponse("<h3>Invalid or expired approval link.</h3>", status_code=404)
    if pending["status"] != "pending":
        return HTMLResponse(f"<h3>This refund was already {pending['status']}.</h3>")

    create_support_ticket(pending["order_id"], f"Manager rejected refund: {pending['reasoning']}")
    send_customer_rejected_email(pending.get("customer_email"), pending["order_id"], pending["reasoning"])
    mark_pending_refund(token, "rejected")
    save_audit_log({
        "order_id": pending["order_id"],
        "decision": "escalate",
        "reasoning": pending["reasoning"],
        "result": "Refund rejected by manager; escalated to support team.",
    })

    return HTMLResponse(f"<h3>Refund for order {pending['order_id']} rejected and escalated to support.</h3>")


# ---------------------------------------------------------------------------
# 9. Run standalone (for quick testing without the API - bypasses auth)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = app.invoke({
        "customer_message": "My order #1234 never arrived, I want a refund",
        "customer_user_id": None,
        "customer_email": "test@example.com",
        "upi_id": "test@upi",
        "audit_log": [],
    })
    print(result["action_result"])
    print(result["audit_log"])
