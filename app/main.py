from langchain_core.tools import tool
from typing_extensions import Annotated
import os
import ag_rules, ag_data
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

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
violations = []
@tool
def rules_agent(query: Annotated[str, "the user query to analyze for rule violations"],
    ) -> list[str]:
    """Detect possible rule violations based on user input."""
    messages = rag_prompt_rules.invoke({"query": query})
    try:
        violations = ag_rules.agent_executor.invoke(messages)
    except Exception as e:
        violations = []
    return violations


template_violation_correction_agent = """
You are an AI assistant responsible for analyzing, correcting, and sanitizing user input while ensuring compliance with rules. 
You have access to the following tools:

1) violation_corrector(input: str, violated_rules: list[str]) → CorrectedOutput
   - Corrects the input text based on the violated rules.
   - Returns a structured output:
     CorrectedOutput(
         corrected_text: str,
         applied_rules: list[str]
     )

2) sanitizer(input: CorrectedOutput) → CorrectedOutput
   - Sanitizes the corrected_text field to remove or anonymize sensitive data using a PII analyzer.
   - Returns a new CorrectedOutput with sanitized text and the same applied_rules.

3) retriever_data(input: CorrectedOutput) → StructuredData
   - Retrieves structured information from the database based on the sanitized corrected text.
   - Returns the extracted structured information.

Behavior:

1. Receive user input and a list of potentially violated rules.  
2. Pass the input and violated rules to `violation_corrector` to generate corrected text.  
3. Take the output of `violation_corrector` and pass it to `sanitizer` to remove or anonymize sensitive data.  
4. Take the sanitized CorrectedOutput and pass it to `retriever_data` to extract structured information.  
5. Return the final output as:
   CorrectedOutput(
       corrected_text=<sanitized corrected text>,
       applied_rules=<rules that were applied>
   )

Additional Guidelines:

- Preserve the original meaning of the input while correcting rule violations.
- Apply only relevant rules listed in violated_rules.
- Ensure all sensitive data is sanitized before further processing.
- The final output must conform to the CorrectedOutput schema.

User Query: {query}
Violated Rules: {violated_rules}

Final Output:
"""
rag_prompt_data = PromptTemplate.from_template(template_rules_agent)

@tool
def data_agent(query: Annotated[str, "the user query to find data"],
    ) -> list[str]:
    """Analyze, correct, and sanitize user input while ensuring compliance with rules."""
    messages = rag_prompt_data.invoke({"query": query})
    try:
        violations = ag_data.agent_executor.invoke(messages)
    except Exception as e:
        violations = []
    return violations

llm_main = init_chat_model("llama3-8b-8192", model_provider="groq")

main_agent_executor = create_react_agent(llm_main, [rules_agent, data_agent])

template_main_agent = """
You are the main AI assistant that coordinates the entire application workflow. 
You have access to two specialized agents:

1) Rules_agent(input: str) → list[str]
   - Analyzes the input text.
   - Returns a list of violated rules (or an empty list if no violations).

2) Data_agent(input: str, rules: list[str]) → CorrectedOutput
   - Takes the original input text and the list of rules returned by Rules_agent.
   - Produces a CorrectedOutput object:
        CorrectedOutput(
            corrected_text: str,   # The corrected and sanitized text
            applied_rules: list[str]  # The rules that were applied for correction
        )

Your Behavior and Workflow:

1. Receive a user query as input.  
2. First, send the query to Rules_agent to determine if there are any violated rules.  
   - If Rules_agent returns an empty list, proceed with the original text.  
   - If Rules_agent returns a list of rules, pass both the input text and the rules to Data_agent.  
3. Data_agent returns a CorrectedOutput object containing corrected_text and applied_rules.  
4. Interpret the CorrectedOutput:  
   - Use corrected_text as the primary version of the user’s input.  
   - Consider applied_rules to understand what was changed and why.  
5. Generate a humanized, natural-language answer to the user, based on corrected_text.  
   - The final output should be written as if you are directly responding to the user’s original request.  
   - Do NOT include metadata, JSON, or schema objects in your response.  
   - Do NOT mention tools, rules, or technical details unless explicitly relevant to the user’s request.  
   - Simply return a fluent, user-facing answer that incorporates the corrections and respects the applied rules.  

Goal:  
- The user should only see a clean, helpful, and natural answer.  
- All technical corrections and sanitization should happen behind the scenes.  

User Query: {query}

Final Answer:
"""
template_main_data = PromptTemplate.from_template(template_main_agent)
