"""
app_v3.py — Union Mobile Customer Support Chatbot (v3)
MLS-4 | Addresses client feedback rounds 1 & 2

Key changes from v2:
  #1  App now invokes the LangGraph telecom_app graph directly — no inline if/else agent logic
  #2  get_greeting() removed; greeting instructions embedded in each agent's system prompt
  #3  All agent prompts enhanced with role guidelines, constraints, tone, and output format
  #4  Agent evaluation node added (Task Completion + Reasoning Coherence, scored 1-5 per turn)

Deployment:
  Local:         streamlit run app_v3.py
  Hugging Face:  push with requirements.txt (see README)
"""

import os
import json
import re
import datetime
from dataclasses import dataclass
from typing import TypedDict, List, Dict, Any

import streamlit as st
import pandas as pd
from openai import OpenAI
from langgraph.graph import StateGraph, END

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Union Mobile — AI Support",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
    color: white; padding: 20px 30px; border-radius: 10px; margin-bottom: 20px;
}
.badge-verified   { background:#4CAF50;color:white;padding:4px 12px;border-radius:20px;font-weight:bold;font-size:.85em; }
.badge-unverified { background:#f44336;color:white;padding:4px 12px;border-radius:20px;font-weight:bold;font-size:.85em; }
.injection-warning { background:#fff3e0;border-left:4px solid #ff6f00;padding:10px 15px;border-radius:4px;margin:10px 0; }
.output-warning    { background:#fce4ec;border-left:4px solid #c62828;padding:10px 15px;border-radius:4px;margin:10px 0; }
.eval-card  { background:#f8f9fa;border:1px solid #dee2e6;padding:12px;border-radius:8px;margin:6px 0; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
MEMORY_FILE = "customer_memory.json"
LARGE_REFUND_THRESHOLD = 50
PLACEHOLDER_NAMES = {"anonymous", "guest", "unknown", "user", "customer", ""}

INJECTION_PATTERNS = [
    r"ignore (all |previous |prior )?(instructions|prompts|rules)",
    r"you are now|pretend (you are|to be)|act as (if you are|a)",
    r"system prompt|reveal (your|the) (prompt|instructions|system)",
    r"jailbreak|dan mode|developer mode|unrestricted mode",
    r"forget (everything|all|prior|previous)",
    r"disregard (all |your |previous )?(instructions|rules|guidelines)",
    r"new persona|override (your|all) (rules|instructions|safety)",
    r"\[system\]|<\|system\|>|##SYSTEM|\{\{system\}\}",
    r"print (your|the) (instructions|prompt|system message)",
    r"bypass (safety|content|filter|restriction)",
]

OUTPUT_SAFETY_PATTERNS = [
    r"(confidential|internal|proprietary) (data|information|details)",
    r"(competitor|rival) (is better|outperforms|superior)",
    r"guaranteed|100% (certain|sure|accurate|correct)",
    r"(sue|lawsuit|legal action) (union mobile|the company)",
    r"(free|no charge|complimentary).{0,30}(forever|permanently|always)",
    r"(your data|customer data|account data) (has been|is being) (sold|shared|leaked)",
]

MANAGER_ONLY_OPERATIONS = [
    "suspend", "cancel", "terminate", "delete account",
    "reset pin", "change owner", "transfer ownership"
]


# ─── UTILITIES ────────────────────────────────────────────────────────────────
def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def scan_for_injection(text: str) -> tuple:
    for p in INJECTION_PATTERNS:
        if re.search(p, text, re.IGNORECASE):
            return True, p
    return False, None

def scan_output_safety(text: str) -> tuple:
    for p in OUTPUT_SAFETY_PATTERNS:
        if re.search(p, text, re.IGNORECASE):
            return True, p
    return False, None

def detect_billing_tier(query: str) -> str:
    q = query.lower()
    if any(kw in q for kw in ["large refund","full refund","waive all","cancel charges","credit entire"]):
        return "manager_only"
    for amt in re.findall(r'\$([\d,]+)', query):
        try:
            if int(amt.replace(',','')) > LARGE_REFUND_THRESHOLD:
                return "manager_only"
        except ValueError:
            pass
    return "standard"


# ─── MEMORY ───────────────────────────────────────────────────────────────────
def load_memory_store() -> dict:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def get_customer_memory(customer_account_id: str, intent_filter: str = None) -> List[dict]:
    store = load_memory_store()
    all_interactions = store.get(customer_account_id, [])
    if not all_interactions:
        return []
    if intent_filter:
        matching = [i for i in all_interactions if i.get('intent') == intent_filter]
        other    = [i for i in all_interactions if i.get('intent') != intent_filter]
        return (matching[-3:] + other[-2:])[-5:]
    return all_interactions[-5:]

def append_customer_memory(customer_account_id: str, interaction: dict) -> None:
    store = load_memory_store()
    if customer_account_id not in store:
        store[customer_account_id] = []
    store[customer_account_id].append(interaction)
    with open(MEMORY_FILE, 'w') as f:
        json.dump(store, f, indent=2)

def format_memory_for_prompt(memory: List[dict]) -> str:
    if not memory:
        return "No previous interactions on record."
    lines = ["=== Customer History (most relevant first) ==="]
    for m in memory:
        lines.append(
            f"[{m.get('timestamp','')[:10]}] {m.get('intent','').upper()} | "
            f"{m.get('agent_used','')} | {m.get('resolution_type','')}\n"
            f"  Query: {m.get('query','')[:100]}\n"
            f"  Summary: {m.get('response_summary','')[:150]}"
        )
    return "\n".join(lines)


# ─── DATASET ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset() -> pd.DataFrame:
    if os.path.exists("df_enriched.csv"):
        return pd.read_csv("df_enriched.csv")
    st.error("df_enriched.csv not found. Please run the notebook first.")
    st.stop()

def retrieve_context(query: str, intent: str, df: pd.DataFrame, n: int = 2) -> str:
    matches = df[df['intent_category'] == intent].copy()
    if matches.empty:
        return "No relevant examples found."
    query_words = set(query.lower().split())
    matches['relevance'] = matches['full_text'].apply(
        lambda t: len(query_words & set(str(t).lower().split()))
    )
    top = matches.nlargest(n, 'relevance')
    parts = [f"[{intent}/{r['resolution_type']}]\n{str(r['full_text'])[:400]}"
             for _, r in top.iterrows()]
    return "\n\n---\n\n".join(parts)


# ─── OPENAI CLIENT ────────────────────────────────────────────────────────────
def _build_client():
    """Build OpenAI client from session state (called inside graph nodes)."""
    api_key  = st.session_state.get("openai_api_key", os.environ.get("OPENAI_API_KEY",""))
    api_base = st.session_state.get("openai_api_base","")
    if api_key and api_base:
        return OpenAI(api_key=api_key, base_url=api_base)
    return OpenAI(api_key=api_key)


# ══════════════════════════════════════════════════════════════════════════════
#  LANGGRAPH STATE + NODE DEFINITIONS
#  All agent logic lives here — the Streamlit app only calls telecom_app.invoke()
# ══════════════════════════════════════════════════════════════════════════════

class GlobalState(TypedDict, total=False):
    customer_name: str
    conversation_id: str
    customer_account_id: str
    verification_status: str
    account_pin_confirmed: bool
    query: str
    conversation_history: List[dict]
    intent_category: str
    injection_flag: bool
    output_flagged: bool
    agent_response: str
    resolution_type: str
    escalation_summary: str
    retrieved_context: str
    customer_memory: List[dict]
    decision_log: List[dict]
    final_response: str
    eval_scores: Dict[str, Any]


# ─── AgentView dataclasses ────────────────────────────────────────────────────
@dataclass
class GuardrailView:
    query: str
    customer_name: str

@dataclass
class IdentityGateView:
    customer_name: str
    conversation_id: str
    customer_account_id: str
    verification_status: str
    account_pin_confirmed: bool
    query: str

@dataclass
class SupervisorView:
    query: str
    customer_name: str
    customer_account_id: str
    verification_status: str
    conversation_history: List[dict]

@dataclass
class NetworkAgentView:
    query: str
    customer_name: str
    verification_status: str
    retrieved_context: str
    conversation_history: List[dict]
    customer_memory: List[dict]

@dataclass
class BillingAgentView:
    query: str
    customer_name: str
    verification_status: str
    billing_tier: str
    retrieved_context: str
    conversation_history: List[dict]
    customer_memory: List[dict]

@dataclass
class AccountAgentView:
    query: str
    customer_name: str
    verification_status: str
    account_pin_confirmed: bool
    retrieved_context: str
    conversation_history: List[dict]
    customer_memory: List[dict]

@dataclass
class EscalationAgentView:
    query: str
    customer_name: str
    customer_account_id: str
    conversation_history: List[dict]
    injection_flag: bool

@dataclass
class OutputGuardrailView:
    agent_response: str
    customer_name: str
    intent_category: str


# ─── NODE 1: INPUT GUARDRAIL ──────────────────────────────────────────────────
def guardrail_node(state: GlobalState) -> GlobalState:
    view = GuardrailView(query=state.get("query",""), customer_name=state.get("customer_name",""))
    flagged, pattern = scan_for_injection(view.query)
    log = {
        "timestamp": utc_now(), "node": "GuardrailNode",
        "customer_name": view.customer_name,
        "verification_status": state.get("verification_status","unverified"),
        "query": view.query[:100], "intent_category": "guardrail",
        "agent_selected": "GuardrailNode", "injection_flag": flagged,
        "resolution_type": "blocked" if flagged else "pass",
        "response_summary": f"Pattern: {pattern}" if flagged else "Clean"
    }
    if flagged:
        safe = "Your request has been flagged for security review. A human agent will assist you shortly."
        return {**state, "injection_flag": True, "agent_response": safe,
                "final_response": safe, "decision_log": state.get("decision_log",[]) + [log]}
    return {**state, "injection_flag": False, "decision_log": state.get("decision_log",[]) + [log]}


# ─── NODE 2: IDENTITY GATE ────────────────────────────────────────────────────
def identity_gate_node(state: GlobalState) -> GlobalState:
    view = IdentityGateView(
        customer_name=state.get("customer_name","Unknown"),
        conversation_id=state.get("conversation_id",""),
        customer_account_id=state.get("customer_account_id",""),
        verification_status=state.get("verification_status","unverified"),
        account_pin_confirmed=state.get("account_pin_confirmed",False),
        query=state.get("query","")
    )
    log = {
        "timestamp": utc_now(), "node": "IdentityGateNode",
        "customer_name": view.customer_name,
        "verification_status": view.verification_status,
        "query": view.query[:100], "intent_category": "identity_check",
        "agent_selected": "IdentityGateNode", "injection_flag": False,
        "resolution_type": "pass" if view.verification_status == "verified" else "restrict",
        "response_summary": f"Status: {view.verification_status}"
    }
    return {**state, "customer_account_id": view.customer_account_id,
            "decision_log": state.get("decision_log",[]) + [log]}


# ─── NODE 3: SUPERVISOR ───────────────────────────────────────────────────────
def supervisor_agent_node(state: GlobalState) -> GlobalState:
    """
    Classifies intent only. Does NOT do RAG (that is delegated to each specialist).
    Loads long-term memory filtered by classified intent.
    """
    oai = _build_client()
    view = SupervisorView(
        query=state.get("query",""),
        customer_name=state.get("customer_name","Unknown"),
        customer_account_id=state.get("customer_account_id",""),
        verification_status=state.get("verification_status","unverified"),
        conversation_history=state.get("conversation_history",[])
    )
    recent = "\n".join(f"{t['role'].upper()}: {t['content'][:100]}"
                       for t in view.conversation_history[-4:])

    system_prompt = """You are the routing supervisor for Union Mobile customer support.
Your ONLY task: classify the customer's intent into exactly one word.

Categories:
  network    — dropped calls, signal, data speed, coverage, outages
  billing    — bill amounts, charges, payments, refunds, invoices, pricing
  account    — plan changes, SIM, suspension, profile updates, PIN changes
  escalation — repeated unresolved complaints, abusive tone, complex multi-department issues

Rules:
- Respond with EXACTLY one word from: network, billing, account, escalation
- Do not explain your reasoning
- When ambiguous, choose the most specific category based on the primary complaint"""

    user_prompt = (f"Customer: {view.customer_name} | Verification: {view.verification_status}\n"
                   f"Recent conversation:\n{recent}\nCurrent query: {view.query}")

    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_prompt}],
        temperature=0, max_tokens=10
    )
    raw    = resp.choices[0].message.content.strip().lower()
    intent = raw if raw in {"network","billing","account","escalation"} else "network"
    memory = get_customer_memory(view.customer_account_id, intent_filter=intent) if view.customer_account_id else []

    log = {
        "timestamp": utc_now(), "node": "SupervisorAgent",
        "customer_name": view.customer_name,
        "verification_status": view.verification_status,
        "query": view.query[:100], "intent_category": intent,
        "agent_selected": f"{intent.capitalize()}Agent",
        "injection_flag": False, "resolution_type": "routing",
        "response_summary": f"Routed to {intent} | memory: {len(memory)} entries"
    }
    return {**state, "intent_category": intent, "customer_memory": memory,
            "retrieved_context": "", "decision_log": state.get("decision_log",[]) + [log]}


# ─── NODE 4: NETWORK AGENT ────────────────────────────────────────────────────
def network_agent_node(state: GlobalState) -> GlobalState:
    """
    Feedback #2: Greeting handled via prompt instruction (no get_greeting() helper).
    Feedback #3: Enhanced system prompt with role, guidelines, constraints, output format.
    """
    oai = _build_client()
    df  = load_dataset()
    view = NetworkAgentView(
        query=state.get("query",""),
        customer_name=state.get("customer_name","Unknown"),
        verification_status=state.get("verification_status","unverified"),
        retrieved_context=retrieve_context(state.get("query",""), "network", df),
        conversation_history=state.get("conversation_history",[]),
        customer_memory=state.get("customer_memory",[])
    )
    history_str = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in view.conversation_history[-6:])
    memory_str  = format_memory_for_prompt(view.customer_memory)

    system_prompt = """You are the Network Support Specialist at Union Mobile — professional, empathetic, and solution-focused.

ROLE & RESPONSIBILITIES:
- Diagnose and resolve connectivity, signal, data speed, and coverage issues
- Guide customers through structured troubleshooting with clear numbered steps
- Escalate to a field technician only when remote resolution is genuinely not possible

GREETING INSTRUCTIONS:
- If the customer name is a real person's name (not "anonymous", "guest", "unknown", or similar placeholders): greet them as "Hello [Name]!"
- If the name is clearly a placeholder or absent: use a neutral "Hello! How can I assist you today?"
- In a multi-turn conversation where you have already greeted the customer: skip the greeting and continue naturally from where you left off

RESPONSE GUIDELINES:
- Acknowledge the inconvenience before providing steps
- Provide exactly 3–5 numbered, specific troubleshooting actions (e.g., "Step 1: Toggle Airplane Mode off/on for 10 seconds")
- Reference the customer's history if relevant (e.g., "I can see you reported a tower issue last week — this may be related")
- Close with a clear next step (e.g., raise a ticket, schedule a technician)

CONSTRAINTS:
- Do not discuss billing or account management — redirect those politely to the correct team
- Do not fabricate network status; if unknown, say so and offer to raise a service ticket
- Never promise a resolution time unless you have confirmed maintenance schedule data

OUTPUT FORMAT:
1. Greeting (first turn only)
2. Empathetic acknowledgement
3. Numbered troubleshooting steps
4. Recommended next step if issue persists"""

    user_prompt = (f"Customer name: {view.customer_name}\n"
                   f"Verification: {view.verification_status}\n\n"
                   f"Conversation history:\n{history_str if history_str else 'None — first message.'}\n\n"
                   f"Customer's interaction history:\n{memory_str}\n\n"
                   f"Knowledge base examples:\n{view.retrieved_context}\n\n"
                   f"Current query: {view.query}")

    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_prompt}],
        temperature=0.3, max_tokens=500
    )
    agent_response = resp.choices[0].message.content.strip()
    log = {
        "timestamp": utc_now(), "node": "NetworkAgent",
        "customer_name": view.customer_name,
        "verification_status": view.verification_status,
        "query": view.query[:100], "intent_category": "network",
        "agent_selected": "NetworkAgent", "injection_flag": False,
        "resolution_type": "troubleshoot", "response_summary": agent_response[:100]
    }
    return {**state, "agent_response": agent_response, "resolution_type": "troubleshoot",
            "retrieved_context": view.retrieved_context,
            "decision_log": state.get("decision_log",[]) + [log]}


# ─── NODE 5: BILLING AGENT ────────────────────────────────────────────────────
def billing_agent_node(state: GlobalState) -> GlobalState:
    """
    Tiered RBAC:
      unverified            → blocked
      verified + large refund → escalated to billing manager
      verified + standard   → full assistance via enhanced prompt
    """
    oai = _build_client()
    df  = load_dataset()
    view = BillingAgentView(
        query=state.get("query",""),
        customer_name=state.get("customer_name","Unknown"),
        verification_status=state.get("verification_status","unverified"),
        billing_tier=detect_billing_tier(state.get("query","")),
        retrieved_context=retrieve_context(state.get("query",""), "billing", df),
        conversation_history=state.get("conversation_history",[]),
        customer_memory=state.get("customer_memory",[])
    )

    # RBAC: unverified
    if view.verification_status != "verified":
        name = view.customer_name.strip()
        greeting = "Hello!" if name.lower() in PLACEHOLDER_NAMES else f"Hello {name}!"
        blocked = (f"{greeting} Billing information is only available to verified account holders. "
                   "Please log in with your name and account PIN using the sidebar.")
        log = {
            "timestamp": utc_now(), "node": "BillingAgent",
            "customer_name": view.customer_name, "verification_status": "unverified",
            "query": view.query[:100], "intent_category": "billing",
            "agent_selected": "BillingAgent", "injection_flag": False,
            "resolution_type": "blocked", "response_summary": "Access denied — unverified"
        }
        return {**state, "agent_response": blocked, "resolution_type": "blocked",
                "decision_log": state.get("decision_log",[]) + [log]}

    # RBAC: large refund → manager
    if view.billing_tier == "manager_only":
        name = view.customer_name.strip()
        greeting = "Hello!" if name.lower() in PLACEHOLDER_NAMES else f"Hello {name}!"
        restricted = (f"{greeting} This refund request exceeds the threshold for standard agent authorisation. "
                      "Refunds above $50 or full charge waivers require a senior billing manager. "
                      "I'm escalating now — a billing manager will contact you within 24 hours.")
        log = {
            "timestamp": utc_now(), "node": "BillingAgent",
            "customer_name": view.customer_name, "verification_status": "verified",
            "query": view.query[:100], "intent_category": "billing",
            "agent_selected": "BillingAgent", "injection_flag": False,
            "resolution_type": "escalate",
            "response_summary": "Large refund escalated to billing manager",
            "AUDIT_FLAG": "LARGE_REFUND_ESCALATED"
        }
        return {**state, "agent_response": restricted, "resolution_type": "escalate",
                "decision_log": state.get("decision_log",[]) + [log]}

    history_str = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in view.conversation_history[-6:])
    memory_str  = format_memory_for_prompt(view.customer_memory)

    system_prompt = """You are the Billing Specialist at Union Mobile — knowledgeable, precise, and empathetic.

ROLE & RESPONSIBILITIES:
- Explain bill charges, billing cycles, plan pricing, and payment options clearly
- Investigate overcharge complaints and process standard refunds (under $50)
- Help customers understand line-item charges and promotional adjustments

GREETING INSTRUCTIONS:
- If the customer name is a real person's name (not "anonymous", "guest", "unknown", or similar): greet them as "Hello [Name]!"
- If the name is a placeholder or absent: use "Hello! How can I assist you today?"
- In a continuing multi-turn conversation: skip the greeting and continue naturally

RESPONSE GUIDELINES:
- Be transparent — explain exactly what each charge represents
- For refunds you process: state clearly "I'll raise a refund request for [amount] — reference: REF-[YYYYMMDD]"
- Cross-reference the customer's history if it provides useful context
- Validate the customer's concern before explaining ("I can see why that would be concerning...")

CONSTRAINTS:
- Do not authorise refunds above $50 — those are automatically escalated to a billing manager
- Do not discuss network technical faults or account suspensions — redirect to those specialist teams
- Never speculate about future charges; only discuss current or past billing periods

OUTPUT FORMAT:
1. Greeting (first turn only)
2. Empathetic acknowledgement of the billing concern
3. Clear explanation of the relevant charge(s) or account status
4. Action taken or next steps (refund reference, escalation note, or further guidance)"""

    user_prompt = (f"Customer name: {view.customer_name}\n"
                   f"Verification: {view.verification_status}\n\n"
                   f"Conversation history:\n{history_str if history_str else 'None — first message.'}\n\n"
                   f"Customer's interaction history:\n{memory_str}\n\n"
                   f"Knowledge base examples:\n{view.retrieved_context}\n\n"
                   f"Current query: {view.query}")

    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_prompt}],
        temperature=0.3, max_tokens=500
    )
    agent_response = resp.choices[0].message.content.strip()
    resolution = "refund" if any(w in agent_response.lower() for w in ["refund","credit","reimburse"]) else "inform"
    log = {
        "timestamp": utc_now(), "node": "BillingAgent",
        "customer_name": view.customer_name, "verification_status": "verified",
        "query": view.query[:100], "intent_category": "billing",
        "agent_selected": "BillingAgent", "injection_flag": False,
        "resolution_type": resolution, "response_summary": agent_response[:100]
    }
    return {**state, "agent_response": agent_response, "resolution_type": resolution,
            "retrieved_context": view.retrieved_context,
            "decision_log": state.get("decision_log",[]) + [log]}


# ─── NODE 6: ACCOUNT AGENT ────────────────────────────────────────────────────
def account_agent_node(state: GlobalState) -> GlobalState:
    oai = _build_client()
    df  = load_dataset()
    view = AccountAgentView(
        query=state.get("query",""),
        customer_name=state.get("customer_name","Unknown"),
        verification_status=state.get("verification_status","unverified"),
        account_pin_confirmed=state.get("account_pin_confirmed",False),
        retrieved_context=retrieve_context(state.get("query",""), "account", df),
        conversation_history=state.get("conversation_history",[]),
        customer_memory=state.get("customer_memory",[])
    )
    name     = view.customer_name.strip()
    greeting = "Hello!" if name.lower() in PLACEHOLDER_NAMES else f"Hello {name}!"

    if view.verification_status != "verified":
        return {**state,
                "agent_response": f"{greeting} Account management requires identity verification. "
                                  "Please log in using the sidebar.",
                "resolution_type": "blocked"}

    if any(op in view.query.lower() for op in MANAGER_ONLY_OPERATIONS) and not view.account_pin_confirmed:
        log = {
            "timestamp": utc_now(), "node": "AccountAgent",
            "customer_name": view.customer_name, "verification_status": "verified",
            "query": view.query[:100], "intent_category": "account",
            "agent_selected": "AccountAgent", "injection_flag": False,
            "resolution_type": "escalate",
            "response_summary": "Sensitive op escalated",
            "AUDIT_FLAG": "SENSITIVE_OPERATION_ATTEMPTED"
        }
        return {**state,
                "agent_response": f"{greeting} This operation requires senior manager authorisation. "
                                  "I'm connecting you with a senior manager now.",
                "resolution_type": "escalate",
                "decision_log": state.get("decision_log",[]) + [log]}

    history_str = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in view.conversation_history[-6:])
    memory_str  = format_memory_for_prompt(view.customer_memory)

    system_prompt = """You are the Account Management Specialist at Union Mobile — organised, clear, and customer-focused.

ROLE & RESPONSIBILITIES:
- Help verified customers change plans, update contact details, manage SIM cards, and adjust account settings
- Guide customers through self-service actions with step-by-step instructions where applicable
- Handle plan upgrade/downgrade requests with accurate billing-cycle and pricing information

GREETING INSTRUCTIONS:
- If the customer name is a real person's name (not "anonymous", "guest", "unknown", or similar): greet them as "Hello [Name]!"
- If the name is a placeholder: use "Hello! How can I assist you today?"
- In a multi-turn conversation: skip the greeting and continue naturally

RESPONSE GUIDELINES:
- For plan changes: clearly state effective date, new pricing, and any contract implications
- For SIM requests: provide step-by-step process or direct to nearest store with address if known
- Reference interaction history to show continuity ("I can see you previously updated your plan in January...")
- Always confirm the action being taken before executing ("I'll proceed with upgrading your plan to Premium Unlimited — shall I confirm?")

CONSTRAINTS:
- MANAGER_ONLY operations (suspend, cancel, terminate, delete account, reset PIN, transfer ownership) require explicit senior authorisation — escalate immediately
- Do not discuss billing charges or network technical issues — those belong to the billing and network teams
- Never confirm irreversible actions without explicit confirmation from the customer

OUTPUT FORMAT:
1. Greeting (first turn only)
2. Confirmation of the requested action
3. Step-by-step instructions or status update
4. Reference number or confirmation of next steps"""

    user_prompt = (f"Customer name: {view.customer_name}\n"
                   f"Verification: {view.verification_status} | PIN confirmed: {view.account_pin_confirmed}\n\n"
                   f"Conversation history:\n{history_str if history_str else 'None — first message.'}\n\n"
                   f"Customer's interaction history:\n{memory_str}\n\n"
                   f"Knowledge base examples:\n{view.retrieved_context}\n\n"
                   f"Current query: {view.query}")

    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_prompt}],
        temperature=0.3, max_tokens=500
    )
    agent_response = resp.choices[0].message.content.strip()
    log = {
        "timestamp": utc_now(), "node": "AccountAgent",
        "customer_name": view.customer_name, "verification_status": "verified",
        "query": view.query[:100], "intent_category": "account",
        "agent_selected": "AccountAgent", "injection_flag": False,
        "resolution_type": "inform", "response_summary": agent_response[:100]
    }
    return {**state, "agent_response": agent_response, "resolution_type": "inform",
            "retrieved_context": view.retrieved_context,
            "decision_log": state.get("decision_log",[]) + [log]}


# ─── NODE 7: ESCALATION AGENT ─────────────────────────────────────────────────
def escalation_agent_node(state: GlobalState) -> GlobalState:
    oai = _build_client()
    view = EscalationAgentView(
        query=state.get("query",""),
        customer_name=state.get("customer_name","Unknown"),
        customer_account_id=state.get("customer_account_id",""),
        conversation_history=state.get("conversation_history",[]),
        injection_flag=state.get("injection_flag",False)
    )
    name        = view.customer_name.strip()
    greeting    = "Hello!" if name.lower() in PLACEHOLDER_NAMES else f"Hello {name}!"
    history_str = "\n".join(f"{t['role'].upper()}: {t['content'][:150]}" for t in view.conversation_history[-6:])

    system_prompt = """You are the Escalation Coordinator at Union Mobile.

ROLE & RESPONSIBILITIES:
- Acknowledge the customer's frustration with genuine empathy
- Create a structured, information-rich handoff packet for the senior specialist team
- Assure the customer their case is being prioritised

RESPONSE TO CUSTOMER:
- Begin with a sincere apology and acknowledgement of their experience
- Confirm the escalation with the assurance that full context is being passed on
- Provide a realistic timeframe (24 hours) without making specific outcome promises

ESCALATION HANDOFF PACKET (internal — structured, concise):
  CUSTOMER: [name and account ID]
  ISSUE: [precise description of the unresolved problem]
  HISTORY: [relevant past interactions from memory]
  ATTEMPTS: [what was tried in this session and why it was insufficient]
  ESCALATION REASON: [why agent-level resolution is not possible]
  URGENCY: [Low / Medium / High with one-line justification]

Keep the handoff packet under 150 words."""

    summary_prompt = (f"Create the escalation handoff packet.\n\n"
                      f"Customer: {view.customer_name} | Account: {view.customer_account_id}\n"
                      f"Current query: {view.query}\n"
                      f"Session history:\n{history_str}")

    sum_resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":summary_prompt}],
        temperature=0.2, max_tokens=300
    )
    esc_summary = sum_resp.choices[0].message.content.strip()

    customer_response = (f"{greeting} I sincerely apologise for the difficulty you've experienced. "
                         "I'm escalating your case to a senior specialist with your full history included. "
                         "You'll receive a personalised follow-up within 24 hours — "
                         "you won't need to repeat yourself.")

    if view.customer_account_id:
        append_customer_memory(view.customer_account_id, {
            "timestamp": utc_now(), "query": view.query[:200], "intent": "escalation",
            "agent_used": "EscalationAgent", "resolution_type": "escalate",
            "response_summary": esc_summary[:300], "escalation_packet": esc_summary
        })

    log = {
        "timestamp": utc_now(), "node": "EscalationAgent_HANDOFF_PACKET",
        "customer_name": view.customer_name,
        "customer_account_id": view.customer_account_id,
        "verification_status": state.get("verification_status","unverified"),
        "query": view.query[:100], "intent_category": "escalation",
        "agent_selected": "EscalationAgent", "injection_flag": view.injection_flag,
        "resolution_type": "escalate",
        "response_summary": customer_response[:100],
        "ESCALATION_SUMMARY": esc_summary
    }
    return {**state, "agent_response": customer_response, "escalation_summary": esc_summary,
            "resolution_type": "escalate",
            "decision_log": state.get("decision_log",[]) + [log]}


# ─── NODE 8: OUTPUT GUARDRAIL ─────────────────────────────────────────────────
def output_guardrail_node(state: GlobalState) -> GlobalState:
    view = OutputGuardrailView(
        agent_response=state.get("agent_response",""),
        customer_name=state.get("customer_name","Unknown"),
        intent_category=state.get("intent_category","")
    )
    flagged, pattern = scan_output_safety(view.agent_response)
    log = {
        "timestamp": utc_now(), "node": "OutputGuardrailNode",
        "customer_name": view.customer_name,
        "verification_status": state.get("verification_status","unverified"),
        "query": state.get("query","")[:100], "intent_category": view.intent_category,
        "agent_selected": "OutputGuardrailNode", "injection_flag": False,
        "output_flagged": flagged,
        "resolution_type": "blocked" if flagged else "pass",
        "response_summary": f"Output violation: {pattern}" if flagged else "Output clean"
    }
    if flagged:
        safe = ("I appreciate your patience. Let me connect you with a specialist "
                "who can provide accurate information for your request.")
        return {**state, "agent_response": safe, "output_flagged": True,
                "decision_log": state.get("decision_log",[]) + [log]}
    return {**state, "output_flagged": False,
            "decision_log": state.get("decision_log",[]) + [log]}


# ─── NODE 9: RESPONSE NODE ────────────────────────────────────────────────────
def response_node(state: GlobalState) -> GlobalState:
    agent_response = state.get("agent_response","I'm unable to process your request at this time.")
    intent   = state.get("intent_category","general")
    acct_id  = state.get("customer_account_id","")
    query    = state.get("query","")
    agent_names = {
        "network":"Network Support Agent","billing":"Billing Specialist",
        "account":"Account Manager","escalation":"Escalation Team","guardrail":"Security System"
    }
    updated_history = state.get("conversation_history",[]).copy()
    updated_history.append({"role":"user","content":query})
    updated_history.append({"role":"assistant","content":agent_response})

    if acct_id and not state.get("injection_flag",False) and intent != "escalation":
        append_customer_memory(acct_id, {
            "timestamp": utc_now(), "query": query[:200], "intent": intent,
            "agent_used": agent_names.get(intent,"Support Agent"),
            "resolution_type": state.get("resolution_type","inform"),
            "response_summary": agent_response[:200]
        })

    log = {
        "timestamp": utc_now(), "node": "ResponseNode",
        "customer_name": state.get("customer_name","Unknown"),
        "verification_status": state.get("verification_status","unverified"),
        "query": query[:100], "intent_category": intent,
        "agent_selected": agent_names.get(intent,"Support Agent"),
        "injection_flag": state.get("injection_flag",False),
        "resolution_type": state.get("resolution_type","inform"),
        "response_summary": agent_response[:100]
    }
    return {**state, "final_response": agent_response,
            "conversation_history": updated_history,
            "decision_log": state.get("decision_log",[]) + [log]}


# ─── NODE 10: EVALUATION NODE (NEW v3) ────────────────────────────────────────
def evaluation_node(state: GlobalState) -> GlobalState:
    """
    Feedback #4 — Per-turn agent evaluation using GPT-4o-mini as evaluator.

    Measures two dimensions:
      Task Completion (1-5)     — did the agent address the customer's actual request?
      Reasoning Coherence (1-5) — is the response logically sound, well-structured, on-policy?

    Scores are stored in state['eval_scores'] for display in the Streamlit evaluation panel.
    Blocked/injection queries are skipped (score: N/A).
    """
    oai = _build_client()
    query    = state.get("query","")
    response = state.get("final_response", state.get("agent_response",""))
    intent   = state.get("intent_category","general")
    ver      = state.get("verification_status","unverified")

    # Skip scoring for blocked/injection cases
    if state.get("injection_flag") or state.get("resolution_type") == "blocked":
        return {**state, "eval_scores": {
            "task_completion":     {"score":"N/A","justification":"Skipped — blocked or injection query"},
            "reasoning_coherence": {"score":"N/A","justification":"Skipped — blocked or injection query"}
        }}

    eval_prompt = f"""You are an impartial evaluator for a telecom customer support AI.
Score the agent's response on two dimensions. Return ONLY valid JSON — no markdown, no explanation.

QUERY: {query}
INTENT CATEGORY: {intent}
CUSTOMER VERIFICATION STATUS: {ver}
AGENT RESPONSE: {response}

SCORING RUBRICS:

TASK COMPLETION — did the agent resolve or meaningfully address the customer's request?
  5 = Fully addressed with actionable and accurate information or next steps
  4 = Mostly addressed, minor gaps remain
  3 = Partially addressed, customer would still have unanswered questions
  2 = Tangentially addressed, customer would likely still be stuck
  1 = Did not address the query at all

REASONING COHERENCE — is the response logically sound, well-structured, and on-policy?
  5 = Excellent structure, correct RBAC applied, fully on-policy, no contradictions
  4 = Good structure with minor inconsistency or slight off-policy phrasing
  3 = Adequate but some logical gaps, missing context, or minor policy violations
  2 = Significant incoherence, incorrect RBAC application, or notable policy deviation
  1 = Incoherent, factually incorrect, or seriously violates policy

Return exactly this JSON:
{{
  "task_completion": {{
    "score": <integer 1-5>,
    "justification": "<one concise sentence>"
  }},
  "reasoning_coherence": {{
    "score": <integer 1-5>,
    "justification": "<one concise sentence>"
  }}
}}"""

    try:
        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":eval_prompt}],
            temperature=0, max_tokens=200
        )
        raw  = re.sub(r"```json|```","", resp.choices[0].message.content.strip())
        eval_scores = json.loads(raw)
    except Exception as e:
        eval_scores = {
            "task_completion":     {"score":"Error","justification":str(e)[:80]},
            "reasoning_coherence": {"score":"Error","justification":str(e)[:80]}
        }
    return {**state, "eval_scores": eval_scores}


# ─── GRAPH ROUTING ────────────────────────────────────────────────────────────
def route_after_guardrail(state: GlobalState) -> str:
    return "end" if state.get("injection_flag",False) else "identity_gate"

def route_supervisor_to_agent(state: GlobalState) -> str:
    return {"network":"network_agent","billing":"billing_agent",
            "account":"account_agent","escalation":"escalation_agent"
            }.get(state.get("intent_category","network"), "network_agent")


# ─── GRAPH COMPILATION ────────────────────────────────────────────────────────
@st.cache_resource
def build_telecom_graph():
    """
    Compile the v3 LangGraph StateGraph.
    Cached so it is only compiled once per Streamlit session.

    v3 flow:
      guardrail → identity_gate → supervisor
        → [network_agent | billing_agent | account_agent | escalation_agent]
        → output_guardrail → response_node → evaluation_node → END
    """
    workflow = StateGraph(GlobalState)

    workflow.add_node("guardrail",        guardrail_node)
    workflow.add_node("identity_gate",    identity_gate_node)
    workflow.add_node("supervisor",       supervisor_agent_node)
    workflow.add_node("network_agent",    network_agent_node)
    workflow.add_node("billing_agent",    billing_agent_node)
    workflow.add_node("account_agent",    account_agent_node)
    workflow.add_node("escalation_agent", escalation_agent_node)
    workflow.add_node("output_guardrail", output_guardrail_node)
    workflow.add_node("response_node",    response_node)
    workflow.add_node("evaluation_node",  evaluation_node)   # NEW v3

    workflow.set_entry_point("guardrail")

    workflow.add_conditional_edges("guardrail", route_after_guardrail,
                                   {"identity_gate":"identity_gate","end":END})
    workflow.add_edge("identity_gate","supervisor")
    workflow.add_conditional_edges("supervisor", route_supervisor_to_agent,
                                   {"network_agent":"network_agent","billing_agent":"billing_agent",
                                    "account_agent":"account_agent","escalation_agent":"escalation_agent"})
    for agent in ["network_agent","billing_agent","account_agent","escalation_agent"]:
        workflow.add_edge(agent,"output_guardrail")
    workflow.add_edge("output_guardrail","response_node")
    workflow.add_edge("response_node","evaluation_node")   # NEW v3
    workflow.add_edge("evaluation_node",END)

    return workflow.compile()


# ─── SESSION STATE ────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "messages":[], "conversation_history":[], "verified":False,
        "customer_name":"", "conversation_id":"", "customer_account_id":"",
        "account_pin_confirmed":False, "decision_log":[],
        "injection_warned":False, "output_warned":False,
        "openai_configured":False, "eval_history":[]
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# Auto-load OpenAI key from HF Spaces secret
if "OPENAI_API_KEY" in os.environ and not st.session_state.get("openai_api_key"):
    st.session_state.openai_api_key  = os.environ["OPENAI_API_KEY"]
    st.session_state.openai_configured = True

df          = load_dataset()
telecom_app = build_telecom_graph()


# ─── PROCESS MESSAGE (single graph invoke call) ───────────────────────────────
def process_message(query: str) -> dict:
    """
    Feedback #1: This function ONLY builds the initial state and calls telecom_app.invoke().
    ALL agent logic, routing, guardrails, memory, evaluation live exclusively in the graph nodes.
    There is no inline if/else agent logic here.
    """
    initial_state = GlobalState(
        query=query,
        customer_name=st.session_state.customer_name or "Guest",
        conversation_id=st.session_state.conversation_id,
        customer_account_id=st.session_state.customer_account_id,
        verification_status="verified" if st.session_state.verified else "unverified",
        account_pin_confirmed=st.session_state.account_pin_confirmed,
        conversation_history=st.session_state.conversation_history.copy(),
        injection_flag=False, output_flagged=False,
        retrieved_context="", customer_memory=[], decision_log=[],
        agent_response="", escalation_summary="",
        resolution_type="", intent_category="", final_response="",
        eval_scores={}
    )

    # ── Single LangGraph invocation — all logic in the graph ──
    result = telecom_app.invoke(initial_state)

    # Sync session state from graph output
    st.session_state.conversation_history = result.get("conversation_history",[])
    st.session_state.decision_log.extend(result.get("decision_log",[]))

    if result.get("injection_flag"):
        st.session_state.injection_warned = True
    if result.get("output_flagged"):
        st.session_state.output_warned = True

    if result.get("eval_scores"):
        st.session_state.eval_history.append({
            "turn":   len(st.session_state.conversation_history) // 2,
            "query":  query[:80],
            "intent": result.get("intent_category",""),
            "scores": result["eval_scores"]
        })

    icons = {"network":"🔧 Network Agent","billing":"💰 Billing Agent",
             "account":"👤 Account Agent","escalation":"🚨 Escalation Team","guardrail":"🛡️ Security System"}
    return {
        "response":       result.get("final_response", result.get("agent_response","")),
        "agent_name":     icons.get(result.get("intent_category",""), "🤖 Support Agent"),
        "intent":         result.get("intent_category",""),
        "resolution_type":result.get("resolution_type",""),
        "injection_flag": result.get("injection_flag",False),
        "output_flagged": result.get("output_flagged",False),
        "eval_scores":    result.get("eval_scores",{})
    }


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
  <h1 style="margin:0;font-size:1.8em;">📱 Union Mobile — AI Customer Support</h1>
  <p style="margin:5px 0 0 0;opacity:.85;">GPT-4o-mini · LangGraph Multi-Agent · Input + Output Guardrails · Per-Turn Evaluation</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    with st.expander("🔑 API Keys", expanded=not st.session_state.openai_configured):
        okey  = st.text_input("OpenAI API Key", type="password",
                               value=st.session_state.get("openai_api_key",""), placeholder="sk-...")
        obase = st.text_input("API Base URL (optional)",
                               value=st.session_state.get("openai_api_base",""),
                               placeholder="Azure/proxy endpoint")
        lskey = st.text_input("LangSmith Key (optional)", type="password",
                               value=st.session_state.get("langsmith_api_key",""))
        if okey:
            st.session_state.openai_api_key = okey
            st.session_state.openai_configured = True
        if obase:
            st.session_state.openai_api_base = obase
        if lskey:
            os.environ.update({"LANGCHAIN_TRACING_V2":"true",
                               "LANGCHAIN_API_KEY":lskey,
                               "LANGCHAIN_PROJECT":"MLS4-Telecom-Chatbot-v3"})

    st.divider()
    st.markdown("## 🔐 Customer Login")
    conv_options = [""] + list(df['conversation_id'].unique())
    sel_conv = st.selectbox("Select Account ID", conv_options)
    if sel_conv:
        st.session_state.conversation_id = sel_conv

    inp_name = st.text_input("Customer Name")
    inp_pin  = st.text_input("Account PIN", type="password")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Verify", type="primary", use_container_width=True):
            if not sel_conv or not inp_name or not inp_pin:
                st.error("Fill all fields")
            else:
                rec = df[df['conversation_id'] == sel_conv]
                if rec.empty:
                    st.error("Account not found")
                else:
                    r = rec.iloc[0]
                    if (inp_name.strip().lower() == str(r['customer_name']).strip().lower()
                            and inp_pin.strip() == str(r['account_pin'])):
                        st.session_state.update({
                            "verified":True,
                            "customer_name":r['customer_name'],
                            "customer_account_id":str(r['customer_account_id']),
                            "account_pin_confirmed":True
                        })
                        st.success("✅ Verified!")
                        st.rerun()
                    else:
                        st.error("Name or PIN incorrect")
    with c2:
        if st.button("Guest", use_container_width=True):
            st.session_state.update({
                "verified":False,
                "customer_name":inp_name or "Guest",
                "customer_account_id":"",
                "account_pin_confirmed":False
            })
            st.rerun()

    st.markdown("### Status")
    if st.session_state.verified:
        st.markdown(f'<span class="badge-verified">✅ VERIFIED — {st.session_state.customer_name}</span>',
                    unsafe_allow_html=True)
        st.caption(f"Account: {st.session_state.customer_account_id}")
    else:
        st.markdown('<span class="badge-unverified">❌ NOT VERIFIED</span>', unsafe_allow_html=True)

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        if st.button("🔄 Reset", use_container_width=True):
            for k in ["messages","conversation_history","verified","customer_name",
                      "conversation_id","customer_account_id","account_pin_confirmed",
                      "decision_log","injection_warned","output_warned","eval_history"]:
                st.session_state.pop(k, None)
            init_session()
            st.rerun()
    with c4:
        if st.session_state.customer_account_id:
            if st.button("🗑️ Clear Mem", use_container_width=True):
                store = load_memory_store()
                store.pop(st.session_state.customer_account_id, None)
                with open(MEMORY_FILE,'w') as f:
                    json.dump(store, f, indent=2)
                st.success("Cleared")

    st.markdown("### 💡 Test Credentials")
    if not df.empty:
        s = df.iloc[0]
        st.code(f"ID:   {s['conversation_id']}\nName: {s['customer_name']}\nPIN:  {s['account_pin']}")


# ── MAIN CHAT AREA ────────────────────────────────────────────────────────────
col_chat, col_info = st.columns([2,1])

with col_chat:
    if st.session_state.injection_warned:
        st.markdown('<div class="injection-warning">⚠️ <b>Input Security Alert:</b> Prompt injection detected and blocked.</div>',
                    unsafe_allow_html=True)
    if st.session_state.output_warned:
        st.markdown('<div class="output-warning">🔴 <b>Output Safety Alert:</b> A generated response was intercepted for policy compliance.</div>',
                    unsafe_allow_html=True)

    st.markdown("### 💬 Chat")

    if not st.session_state.messages:
        welcome = (f"Hello {st.session_state.customer_name}! I'm your Union Mobile AI support assistant. "
                   "I can help with network issues, billing questions, and account management."
                   if st.session_state.verified else
                   "Welcome to Union Mobile support! For billing and account access, please verify your identity in the sidebar.")
        st.session_state.messages.append({"role":"assistant","content":welcome,
                                           "agent":"🤖 Support Assistant","timestamp":utc_now()})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("agent"):
                st.caption(msg["agent"])
            st.write(msg["content"])

    if user_input := st.chat_input("Type your message here..."):
        if not st.session_state.get("openai_api_key") and "OPENAI_API_KEY" not in os.environ:
            st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
        else:
            st.session_state.messages.append({"role":"user","content":user_input,"timestamp":utc_now()})
            with st.spinner("Routing to the right agent..."):
                try:
                    result = process_message(user_input)
                    st.session_state.messages.append({
                        "role":"assistant","content":result["response"],
                        "agent":result["agent_name"],"timestamp":utc_now()
                    })
                except Exception as e:
                    st.session_state.messages.append({
                        "role":"assistant",
                        "content":f"Technical issue, please try again. ({str(e)[:80]})",
                        "agent":"⚠️ System","timestamp":utc_now()
                    })
            st.rerun()


with col_info:
    st.markdown("### 📋 Interaction History")
    with st.expander("Past Interactions", expanded=False):
        acct = st.session_state.customer_account_id
        if acct:
            memory = get_customer_memory(acct)
            if memory:
                for m in reversed(memory):
                    ts = m.get('timestamp','')[:10]
                    if m.get('escalation_packet'):
                        st.markdown(f"**{ts}** — 🚨 ESCALATION")
                        st.text_area("Handoff Packet", m['escalation_packet'], height=90, disabled=True)
                    else:
                        st.markdown(f"**{ts}** — _{m.get('intent','').upper()}_\n"
                                    f"- 🤖 {m.get('agent_used','')}\n"
                                    f"- ✓ {m.get('resolution_type','')}\n"
                                    f"- 💬 _{m.get('query','')[:60]}..._\n---")
            else:
                st.info("No past interactions.")
        else:
            st.info("Verify your identity to view history.")

    if st.session_state.verified and st.session_state.conversation_id:
        st.markdown("### 👤 Account")
        rec = df[df['conversation_id'] == st.session_state.conversation_id]
        if not rec.empty:
            r = rec.iloc[0]
            with st.expander("Details", expanded=True):
                st.write(f"**Name:** {r['customer_name']}")
                st.write(f"**Account ID:** {r.get('customer_account_id','N/A')}")
                st.write(f"**Access Level:** {r['access_level']}")

    st.markdown("### ⚡ Quick Actions")
    actions = {
        "📶 Signal issue": "My signal keeps dropping. What troubleshooting steps can I take?",
        "💳 Check bill":   "Can you explain why my bill is higher than usual this month?",
        "📱 Change plan":  "I'd like to upgrade to a plan with more data.",
        "🆘 Get help":     "I've had this issue unresolved for three weeks now."
    }
    for label, msg in actions.items():
        if st.button(label, use_container_width=True):
            if not st.session_state.get("openai_api_key") and "OPENAI_API_KEY" not in os.environ:
                st.error("Add OpenAI API key first")
            else:
                st.session_state.messages.append({"role":"user","content":msg,"timestamp":utc_now()})
                with st.spinner("Processing..."):
                    try:
                        result = process_message(msg)
                        st.session_state.messages.append({
                            "role":"assistant","content":result["response"],
                            "agent":result["agent_name"],"timestamp":utc_now()
                        })
                    except Exception as e:
                        st.error(str(e)[:80])
                st.rerun()

    turns = len(st.session_state.conversation_history) // 2
    if turns > 0:
        st.markdown("### 🔄 Session")
        st.metric("Turns", turns)
        st.caption("Full history carried forward each turn.")


# ── TABS: AUDIT LOG + EVALUATION ──────────────────────────────────────────────
st.divider()
tab_audit, tab_eval = st.tabs(["📊 Decision Log (Audit Trail)", "🎯 Agent Evaluation"])

with tab_audit:
    if st.session_state.decision_log:
        COLORS = {
            "network":"#E3F2FD","billing":"#E8F5E9","account":"#FFF3E0",
            "escalation":"#FFEBEE","guardrail":"#F3E5F5",
            "identity_check":"#E0F2F1","routing":"#F5F5F5"
        }
        df_log = pd.DataFrame(st.session_state.decision_log)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Nodes", len(df_log))
        c2.metric("Injections Blocked",
                  int(df_log['injection_flag'].sum()) if 'injection_flag' in df_log else 0)
        c3.metric("Output Intercepts",
                  int(df_log['output_flagged'].sum()) if 'output_flagged' in df_log else 0)
        c4.metric("Escalations",
                  len(df_log[df_log['resolution_type']=='escalate']) if 'resolution_type' in df_log else 0)

        show_cols = ['timestamp','node','customer_name','verification_status',
                     'intent_category','injection_flag','resolution_type','response_summary']
        show = [c for c in show_cols if c in df_log.columns]

        def color_row(row):
            bg = COLORS.get(row.get('intent_category',''),'#FFFFFF')
            return [f'background-color:{bg}']*len(row)

        st.dataframe(df_log[show].style.apply(color_row,axis=1),
                     use_container_width=True, height=300)
        st.download_button("⬇️ Download Audit CSV",
                           data=df_log[show].to_csv(index=False),
                           file_name=f"audit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")
        st.caption("🔵 Network · 🟢 Billing · 🟠 Account · 🔴 Escalation · 🟣 Guardrail")
    else:
        st.info("No log entries yet. Start a conversation.")


with tab_eval:
    st.markdown("### 🎯 Per-Turn Agent Evaluation")
    st.caption("Scores auto-generated after each turn by GPT-4o-mini acting as an independent evaluator.")

    if st.session_state.eval_history:
        tc_scores = [e['scores'].get('task_completion',{}).get('score')
                     for e in st.session_state.eval_history
                     if isinstance(e['scores'].get('task_completion',{}).get('score'),int)]
        rc_scores = [e['scores'].get('reasoning_coherence',{}).get('score')
                     for e in st.session_state.eval_history
                     if isinstance(e['scores'].get('reasoning_coherence',{}).get('score'),int)]

        m1,m2,m3 = st.columns(3)
        m1.metric("Turns Evaluated", len(st.session_state.eval_history))
        m2.metric("Avg Task Completion",
                  f"{sum(tc_scores)/len(tc_scores):.1f}/5" if tc_scores else "N/A")
        m3.metric("Avg Reasoning Coherence",
                  f"{sum(rc_scores)/len(rc_scores):.1f}/5" if rc_scores else "N/A")

        if tc_scores and rc_scores:
            chart_data = pd.DataFrame({
                "Turn": list(range(1,len(tc_scores)+1)),
                "Task Completion": tc_scores,
                "Reasoning Coherence": rc_scores
            }).set_index("Turn")
            st.line_chart(chart_data, use_container_width=True, height=180)

        st.divider()
        for entry in reversed(st.session_state.eval_history):
            scores = entry.get('scores',{})
            tc = scores.get('task_completion',{})
            rc = scores.get('reasoning_coherence',{})
            tc_s = tc.get('score','N/A')
            rc_s = rc.get('score','N/A')
            def col(s): return "🟢" if isinstance(s,int) and s>=4 else ("🟡" if isinstance(s,int) and s==3 else "🔴")
            with st.expander(
                f"Turn {entry['turn']} · {entry['intent'].upper()} · "
                f"TC: {col(tc_s)} {tc_s}/5 · RC: {col(rc_s)} {rc_s}/5",
                expanded=False
            ):
                st.markdown(f"**Query:** _{entry['query']}_")
                cx,cy = st.columns(2)
                with cx:
                    st.markdown(f"**Task Completion: {col(tc_s)} {tc_s}/5**")
                    st.caption(tc.get('justification',''))
                with cy:
                    st.markdown(f"**Reasoning Coherence: {col(rc_s)} {rc_s}/5**")
                    st.caption(rc.get('justification',''))
    else:
        st.info("No evaluations yet. Start a conversation to see scores here.")
        st.markdown("""
**What these metrics measure:**

**Task Completion (1–5):** Did the agent fully address what the customer asked?
5 = complete actionable answer · 1 = missed the question entirely

**Reasoning Coherence (1–5):** Is the response logically sound, well-structured, and on-policy?
5 = correct RBAC, clear structure, no contradictions · 1 = incoherent or policy-violating
""")


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#888;font-size:.8em;padding:10px">
  Union Mobile AI Support — MLS-4 v3 | LangGraph · Input+Output Guardrails · Multi-Turn · Per-Turn Evaluation
  <br>⚠️ Demonstration system only.
</div>
""", unsafe_allow_html=True)
