# ResolveAI — Autonomous Customer Support Agent

**TakeOver'26 Hackathon Submission**
Theme: AI Automation & Intelligent Agents

## Overview

ResolveAI is an autonomous customer support agent that goes beyond a typical chatbot — instead of just answering questions, it reasons over real order data and takes real actions: processing refunds, escalating to human agents, or answering general policy questions, all without manual intervention.

## Live Links

- **Backend API:** https://resolveai-hackathon.onrender.com
- **API Docs (Swagger UI):** https://resolveai-hackathon.onrender.com/docs

## The Problem

Businesses can't provide instant responses to customer queries, resulting in delayed support and reduced customer satisfaction. Most existing solutions are simple chatbots that only generate text responses — they don't actually resolve anything.

## The Solution

ResolveAI is a true autonomous agent built with LangGraph that:

1. **Understands** the customer's message and classifies its intent (order issue, general question, greeting, or unclear)
2. **Retrieves context** — pulls real order data from a live database and relevant policy information
3. **Reasons and decides** — applies business policy logic to determine the right outcome
4. **Takes real action** — updates order records, creates support tickets, or answers directly, not just simulated text
5. **Logs everything** — every decision is recorded with its reasoning for full transparency and auditability

## Key Differentiator

> "We didn't build a chatbot that talks about your order — we built an agent that fixes it."

Unlike a typical support bot, ResolveAI actually writes to a database, makes autonomous decisions with real business logic, and knows when to defer to a human — all visible through a transparent audit trail.

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| LLM | Google Gemini 2.5 Flash |
| Backend API | FastAPI |
| Database | Supabase (PostgreSQL) |
| Hosting | Render |

## Architecture

Customer Message
↓
Understand & Classify
↓
Retrieve Context (Supabase + Policy)
↓
Reason & Decide ──┬──→ Auto-Resolve (refund processed)
├──→ Escalate (ticket created for human review)
├──→ Needs Info (asks for order ID)
└──→ Answered (general question / greeting)
↓
Audit Trail (logged to database)

## API Usage

**Endpoint:** `POST /chat`

**Request:**
```json
{
  "customer_message": "My order #1234 never arrived, I want a refund"
}
```

**Response:**
```json
{
  "decision": "auto_resolve",
  "reasoning": "The order is 9 days late, which is more than 7 days, and the customer has 0 refunds this month.",
  "action_result": "Refund processed for order 1234",
  "audit_log": [...]
}
```

## Decision Types

| Decision | Meaning |
|---|---|
| `auto_resolve` | Agent autonomously processed a refund and updated records |
| `escalate` | Agent created a support ticket for human review (e.g. excessive refund requests) |
| `needs_info` | Agent needs an order ID to proceed |
| `answered` | Agent answered a general question, greeting, or unclear message directly |

## Running Locally

```bash
pip install -r requirements.txt
uvicorn support_agent_graph:app_api --reload --port 8000
```

Requires a `.env` file with:

SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
GOOGLE_API_KEY=your_gemini_api_key_here

## Team

- Chandra Mohan Vamshi — Backend (LangGraph, FastAPI, Gemini, Supabase)
- [Sanketh ,Saketh ,Sai Rishith ] — Frontend (React)

Built for TakeOver'26 Hackathon.
