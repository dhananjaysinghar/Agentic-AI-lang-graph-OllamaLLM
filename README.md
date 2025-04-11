# Agentic-AI-lang-graph-OllamaLLM
Agentic-AI-lang-graph-OllamaLLM


# LangGraph Workflow Diagram

```mermaid
---
config:
  flowchart:
    curve: linear
---

graph TD;
	__start__([<p>__start__</p>]):::first
	reformulate_question(reformulate_question)
	generate_answer(generate_answer)
	fact_check_agent(fact_check_agent)
	summarize(summarize)
	__end__([<p>__end__</p>]):::last
	__start__ --> reformulate_question;
	fact_check_agent --> summarize;
	generate_answer --> fact_check_agent;
	reformulate_question --> generate_answer;
	summarize --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```



![image](https://github.com/user-attachments/assets/9951df97-c65d-4734-8593-8d661f197ef3)


## Langgraph integration
```
https://ollama.com/download/
ollama run llama3.2
ollama pull mistral
ollama run mistral

pip install langchain
pip install langchain_community
pip install langgraph
pip install langchain-ollama --upgrade
```
