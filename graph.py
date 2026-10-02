import os
import logging
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from ingestion import PipelineConfig
from retrieval import search_policies

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CopilotState(TypedDict):
    complaint: str
    retrieved_context: str
    draft_response: str
    compliance_feedback: str
    is_approved: bool

def retrieve_node(state: CopilotState) -> dict:
    logger.info("--- NODE: RETRIEVE ---")
    query = state["complaint"]
    cfg = PipelineConfig()
    found_docs = search_policies(query=query, config=cfg, top_k=2)

    context_str = "\n\n".join(
        [f"[Source: {doc.metadata.get('source')}]\n{doc.page_content}" for doc, score in found_docs]
    )
    return {"retrieved_context": context_str}


def draft_node(state: CopilotState) -> dict:
    logger.info("--- NODE: DRAFT ---")
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.0)

    feedback_section = ""
    if state.get("compliance_feedback"):
        feedback_section = f"\n\nCRITIC FEEDBACK TO FIX: {state['compliance_feedback']}"

    system_prompt = """You are a professional legal assistant.
    Answer the complaint using ONLY the provided policy context.
    Do not invent information. Be empathetic but legally strict.

    CONTEXT:
    {context}
    {feedback}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Complaint: {complaint}")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "context": state["retrieved_context"],
        "complaint": state["complaint"],
        "feedback": feedback_section
    })

    return {"draft_response": response.content}


class CriticOutput(BaseModel):
    is_approved: bool = Field(
        description="True if the draft strictly follows the context and is professional, False otherwise.")
    feedback: str = Field(
        description="If not approved, explain exactly what the drafter must fix. If approved, leave empty.")


def critic_node(state: CopilotState) -> dict:
    logger.info("--- NODE: COMPLIANCE CRITIC ---")
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.0)
    evaluator = llm.with_structured_output(CriticOutput)

    evaluation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict Compliance Officer. 
        Compare the Draft Response against the Corporate Context.
        RULES:
        1. The draft MUST NOT promise refunds if the context forbids it.
        2. The draft MUST NOT contain information outside the context.
        If it violates rules, return is_approved=False and provide actionable feedback."""),
        ("human", "CONTEXT:\n{context}\n\nDRAFT RESPONSE:\n{draft}")
    ])

    chain = evaluation_prompt | evaluator

    result: CriticOutput = chain.invoke({
        "context": state["retrieved_context"],
        "draft": state["draft_response"]
    })

    logger.info(f"Critic Verdict: Approved={result.is_approved}")
    if not result.is_approved:
        logger.warning(f"Critic Feedback: {result.feedback}")

    return {
        "is_approved": result.is_approved,
        "compliance_feedback": result.feedback
    }

def route_compliance(state: CopilotState) -> Literal["end", "drafter"]:
    """
    Перевіряє State. Якщо is_approved == True, направляє граф на вихід.
    Якщо False — повертає назад до Drafter Node.
    """
    if state.get("is_approved"):
        return "end"
    return "drafter"

def build_graph() -> StateGraph:
    workflow = StateGraph(CopilotState)

    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("drafter", draft_node)
    workflow.add_node("critic", critic_node)

    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "drafter")

    workflow.add_edge("drafter", "critic")

    workflow.add_conditional_edges(
        "critic",
        route_compliance,
        {
            "end": END,
            "drafter": "drafter"
        }
    )

    return workflow.compile()


if __name__ == "__main__":
    app = build_graph()

    initial_state = {
        "complaint": "I was scammed over the phone and transferred 500 dollars to a fraudster. I need a refund immediately!",
    }

    final_state = app.invoke(initial_state)

    print(final_state["draft_response"])
    print("=" * 50)