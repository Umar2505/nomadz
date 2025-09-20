import os
import getpass

# Option A: Prompt for secret
if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass(
        "Enter LangChain/LangSmith API key:"
    )

# # 1️⃣ Set the Groq key before any LangChain import
# if not os.environ.get("GROQ_API_KEY"):
#     os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq:")

from tools import (
    retriever_data,
    sanitizer,
    violation_corrector,
    average,
    total,
    less_than,
    greater_than,
)

from langchain.chat_models import init_chat_model

# from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# if not os.environ.get("GROQ_API_KEY"):
#     os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("allam-2-7b", model_provider="groq")

tools = [
    violation_corrector,
    retriever_data,
    sanitizer,
    average,
    total,
    less_than,
    greater_than,
]
agent_executor = create_react_agent(model=model, tools=tools)

