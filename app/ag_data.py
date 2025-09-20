from tools import retriever_data, sanitizer, violation_corrector
import getpass
import os
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("allam-2-7b", model_provider="groq")

tools = [violation_corrector, retriever_data, sanitizer]
agent_executor = create_react_agent(llm=model, tools=tools)