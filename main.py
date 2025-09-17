from __future__ import annotations

import os
from pathlib import Path
from typing import List

import yaml
from langsmith.run_helpers import tracing_context  # <- LangSmith top-level run context

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langchain_openai import ChatOpenAI

# Tools
from tools.tools import TOOLS

# export OPENROUTER_API_KEY="sk-or-..."
# export LANGSMITH_API_KEY="lsv2_..."        # required for tracing
# export LANGSMITH_TRACING_V2=true           # turn on v2 tracing
# export LANGSMITH_PROJECT="openrouter-langgraph-demo-4324"
# export LANGSMITH_WORKSPACE_ID="..."      # if you need to target a specific workspace



ROOT = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

# ---- LangSmith v2 tracing (env) ----
# Required: LANGSMITH_API_KEY; Recommended: LANGSMITH_PROJECT
os.environ.setdefault("LANGSMITH_TRACING_V2", "true")  # v2 tracing switch
if "langsmith_project" in CONFIG:
    os.environ["LANGSMITH_PROJECT"] = CONFIG["langsmith_project"]
# (Optional) if you have multiple workspaces:
# os.environ["LANGSMITH_WORKSPACE_ID"] = "<workspace-id>"  # 

# ---- Model via OpenRouter (OpenAI-compatible) ----
llm = ChatOpenAI(
    model=CONFIG["model"],
    temperature=CONFIG.get("temperature", 0),
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url=CONFIG.get("openrouter_base_url", "https://openrouter.ai/api/v1"),  # 
)

# Advertise tools to the model; (optional) name the run for clarity in LangSmith UI
llm_with_tools = llm.bind_tools(TOOLS).with_config({"run_name": "chat_model+tools"})

# ---- Graph: model ↔ tools loop ----
def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

def call_model(state: MessagesState, *, config=None):
    messages: List[AnyMessage] = state["messages"]
    ai = llm_with_tools.invoke(messages, config=config)
    return {"messages": [ai]}

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(TOOLS).with_config({"run_name": "tool_node"}))  # tools traced individually
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_continue, ["tools", END])
builder.add_edge("tools", "call_model")

checkpointer = InMemorySaver()  # preserves threaded history and tool messages 
graph = builder.compile(checkpointer=checkpointer)

# ---- Demo run ----
def _system_prompt() -> str:
    p = ROOT / CONFIG["paths"]["system_prompt"]
    return p.read_text(encoding="utf-8")

def _sample_prompt() -> str:
    return "Use the coherence check tool with a made up sample to test it"

if __name__ == "__main__":
    system = SystemMessage(_system_prompt())
    user = HumanMessage(_sample_prompt())
    thread_id = CONFIG.get("thread_id", "demo-thread-001")
    cfg = {"configurable": {"thread_id": thread_id}, "tags": ["demo", "langgraph", "openrouter"]}

    # Top-level LangSmith run (adds clear name/metadata in the UI)
    with tracing_context(name="agent_run", metadata={"thread_id": thread_id}, tags=["agent", "tools"]):
        final = graph.invoke({"messages": [system, user]}, config=cfg)
        print("\nAssistant:", final["messages"][-1].content)

        followup = HumanMessage("Great. Now show only titles for the blue items.")
        final2 = graph.invoke({"messages": [followup]}, config=cfg)
        print("\nAssistant (follow-up):", final2["messages"][-1].content)
