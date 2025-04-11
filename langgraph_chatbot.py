import chainlit as cl
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
from typing import TypedDict
from textwrap import dedent

# 1. Shared state
class QAState(TypedDict):
    question: str
    reformulated_question: str
    answer: str
    fact_check: str
    summary: str

# 2. LLM instances
llm_base = OllamaLLM(model="mistral")
llm_streaming = OllamaLLM(model="mistral", streaming=True)

# 3. Helper to send messages
async def send_msg(label: str, content: str, author: str = "🧠 Agent"):
    await cl.Message(content=f"{label} {content.strip()}", author=author).send()

# 4. Graph Nodes
async def reformulate_question_node(state: QAState) -> QAState:
    q = state["question"]
    new_q = await llm_base.ainvoke(f"Rephrase this question clearly: {q}")
    await send_msg("🔄 Reformulated:", new_q)
    return {**state, "reformulated_question": new_q.strip()}

async def generate_answer_node(state: QAState) -> QAState:
    q = state["reformulated_question"]
    msg = cl.Message(content="", author="📘 Answer Agent")
    await msg.send()

    full_answer = ""
    async for token in llm_streaming.astream(q):
        full_answer += token
        await msg.stream_token(token)

    msg.content = full_answer
    await msg.update()
    return {**state, "answer": full_answer.strip()}

async def fact_check_agent(state: QAState) -> QAState:
    prompt = dedent(f"""
        You are a fact-checking agent. Verify the accuracy of this answer.
        Question: {state["reformulated_question"]}
        Answer: {state["answer"]}
        Provide a verdict and cite sources if possible.
    """)
    check = await llm_base.ainvoke(prompt)
    await send_msg("🔍 Fact Check:", check)
    return {**state, "fact_check": check.strip()}

async def summarize_node(state: QAState) -> QAState:
    prompt = dedent(f"""
        Summarize the final answer with fact-checking included:
        Answer: {state["answer"]}
        Fact-Check: {state["fact_check"]}
    """)
    summary = await llm_base.ainvoke(prompt)
    await send_msg("📝 Summary:", summary)
    return {**state, "summary": summary.strip()}

# 5. Build LangGraph
builder = StateGraph(QAState)
builder.set_entry_point("reformulate_question")
builder.add_node("reformulate_question", reformulate_question_node)
builder.add_node("generate_answer", generate_answer_node)
builder.add_node("fact_check_agent", fact_check_agent)
builder.add_node("summarize", summarize_node)

builder.add_edge("reformulate_question", "generate_answer")
builder.add_edge("generate_answer", "fact_check_agent")
builder.add_edge("fact_check_agent", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
# with open("graph.mmd", "w") as f:
#     f.write(graph.get_graph().draw_mermaid())

# 6. Chainlit entry
@cl.on_message
async def on_message(message: cl.Message):
    await cl.Message(content=f"❓ Question: {message.content}").send()
    await graph.ainvoke({"question": message.content})

# chainlit run /Users/dhananjayasamantasinghar/Desktop/test-python/src/test/langgraph_chatbot.py
