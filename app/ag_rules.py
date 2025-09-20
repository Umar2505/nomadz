from IPython.display import Image, display
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
import tools

llm_ruler = init_chat_model("llama3-8b-8192", model_provider="groq")

agent_executor = create_react_agent(llm_ruler, [tools.parser, tools.retriever_rules])

display(Image(agent_executor.get_graph().draw_mermaid_png()))