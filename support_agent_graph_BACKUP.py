"""
ResolveAI - Autonomous Customer Support Agent
LangGraph workflow: Understand -> Retrieve Context -> Reason & Decide
                     -> (Auto-resolve | Escalate) -> Audit Trail

Install:
    pip install langgraph langchain-google-genai supabase fastapi uvicorn --break-system-packages

Run as API:
    uvicorn support_agent_graph_supabase:app_api --reload --port 8000
"""

from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from supabase import create_client
import json
import re

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# ---------------------------------------------------------------------------
# Supabase setup - replace with your actual project URL and publishable key
# ---------------------------------------------------------------------------
import os
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# 1. Shared state that flows through every node
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    customer_message: str
    intent: Optional[str]
    order_id: Optional[str]
    order_data: Optional[dict]
    policy_context: Optional[str]
    decision: Optional[str]          # "auto_resolve" | "escalate"
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
    prompt = f"""Extract the customer's intent and any order ID from this message.
Return ONLY JSON, no other text: {{"intent": "...", "order_id": "..." or null}}

Message: {state['customer_message']}"""

    try:
        response = llm.invoke(prompt)
        parsed = json.loads(clean_json(response.content))
        intent = parsed.get("intent", "unknown")
        order_id = parsed.get("order_id")
    except (json.JSONDecodeError, KeyError, AttributeError):
        intent = "unknown"
        order_id = None

    return {**state, "intent": intent, "order_id": order_id}


# ---------------------------------------------------------------------------
# 3. Node: Retrieve context (Supabase lookup + policy RAG)
# ---------------------------------------------------------------------------
def retrieve_context(state: AgentState) -> AgentState:
    order_data = fetch_order_from_db(state["order_id"])
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
            "reasoning": f"No order found matching ID {state['order_id']}.",
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


def route_decision(state: AgentState) -> Literal["auto_resolve", "escalate", "needs_info"]:
    """Conditional edge function - reads state, returns next node name."""
    if state["decision"] in ("auto_resolve", "escalate", "needs_info"):
        return state["decision"]
    return "escalate"


# ---------------------------------------------------------------------------
# 4b. Node: Needs info (ask the customer for their order ID)
# ---------------------------------------------------------------------------
def needs_info(state: AgentState) -> AgentState:
    if state["order_id"]:
        message = f"I couldn't find an order matching ID {state['order_id']}. Could you double-check the order number?"
    else:
        message = "Could you please share your order ID so I can look into this for you?"

    return {**state, "action_result": message}


# ---------------------------------------------------------------------------
# 5a. Node: Auto-resolve (real write action)
# ---------------------------------------------------------------------------
def auto_resolve(state: AgentState) -> AgentState:
    update_order_status_in_db(state["order_id"], status="refunded")
    send_confirmation_email(state["order_id"])

    return {
        **state,
        "action_result": f"Refund processed for order {state['order_id']}",
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
# 6. Node: Audit trail (both branches merge here)
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
    },
)

graph.add_edge("auto_resolve", "audit_trail")
graph.add_edge("escalate", "audit_trail")
graph.add_edge("needs_info", END)
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


def fetch_order_from_db(order_id):
    if not order_id:
        return None
    try:
        response = supabase.table("orders").select("*").eq("order_id", order_id).execute()
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


def send_confirmation_email(order_id):
    print(f"[EMAIL] Confirmation sent to customer for order {order_id}")


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


# ---------------------------------------------------------------------------
# 8. FastAPI wrapper - exposes the agent as a web API
# ---------------------------------------------------------------------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app_api = FastAPI()

app_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageRequest(BaseModel):
    customer_message: str


@app_api.post("/chat")
def chat(request: MessageRequest):
    try:
        result = app.invoke({
            "customer_message": request.customer_message,
            "audit_log": [],
        })
        return {
            "decision": result["decision"],
            "reasoning": result["reasoning"],
            "action_result": result["action_result"],
            "audit_log": result["audit_log"],
        }
    except Exception as e:
        return {"error": "Something went wrong processing this request.", "detail": str(e)}


# ---------------------------------------------------------------------------
# 9. Run standalone (for quick testing without the API)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = app.invoke({
        "customer_message": "My order #1234 never arrived, I want a refund",
        "audit_log": [],
    })
    print(result["action_result"])
    print(result["audit_log"])
