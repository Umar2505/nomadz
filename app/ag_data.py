from tools import retriever_data, sanitizer
import getpass
import os

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("allam-2-7b", model_provider="groq")

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Create the agent
memory = MemorySaver()
retrieve = retriever_data()
sanitize = sanitizer()
tools = [retrieve, sanitize]
# tools = []
agent_executor = create_react_agent(model, tools, checkpointer=memory)

