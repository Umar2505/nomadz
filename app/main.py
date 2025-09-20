import os
import ag_rules
from langchain_core.prompts import PromptTemplate

os.environ["LANGSMITH_TRACING"] = "true"


template_rules_agent = """You are an AI assistant that detects possible rule violations based on user input. 
You have access to two tools:

1) check(input: str) → list[str]  
   - This tool analyzes the input string and returns a list of possible keywords relevant to rule violations.  
   - If no keywords are found, it returns an empty list.  

2) retrieve(keywords: list[str]) → list[str]  
   - This tool retrieves possible rule violations from the vector store.  
   - It takes a list of keywords and returns a list of possible rule violations.

Behavior:
- When given a user query, first call `check()` with the input string.  
- If `check()` returns an empty list, return an empty list as the final answer.  
- If `check()` returns keywords, pass them to `retrieve()`.  
- Return the list of possible rule violations retrieved.  

Query: {query}  
Answer:
"""
rag_prompt_rules = PromptTemplate.from_template(template_rules_agent)

