import os
from IPython.display import Image, display
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import Annotated, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

os.environ["LANGSMITH_TRACING"] = "true"

# Load and split PDF document
file_path1 = "CELEX.pdf"
loader1 = PyPDFLoader(file_path1)

rule = loader1.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
rules = text_splitter.split_documents(rule)

# Load and split CSV document
file_path2 = "patients.csv"
loader2 = CSVLoader(file_path2, metadata_columns=["Id", "BIRTHDATE", "DEATHDATE", "SSN", "DRIVERS", "PASSPORT", "PREFIX", "FIRST", "LAST", "SUFFIX", "MAIDEN", "MARITAL", "RACE", "ETHNICITY", "GENDER", "BIRTHPLACE", "ADDRESS", "CITY", "STATE", "COUNTY", "ZIP", "LAT", "LON", "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE"])
data = loader2.load()

# Initialize memory saver
memory = MemorySaver()

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create vector store and add rules
vector_store1 = InMemoryVectorStore(embeddings)
rul = vector_store1.add_documents(rules)

vector_store2 = InMemoryVectorStore(embeddings)
dat = vector_store2.add_documents(data)

template = """You are an AI assistant that finds keywords from the query parts that can cause rule violations to retrieve relevant rules and regulations from another tool. You should return a list of keywords that are relevant to the query separated by commas. If there are no relevant keywords, return an empty list.

Query: {query}
Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

@tool
def check(
    input: Annotated[str, "the query to analyze for rule keywords"],
    ) -> list[str]:
    """Analyze the query for possible rule violation keywords."""
    messages = custom_rag_prompt.invoke({"query": input})
    try:
        keywords = ",".split(llm.invoke(messages))
    except Exception as e:
        keywords = []
    return keywords

@tool(response_format="content_and_artifact")
def retrieve(
    keywords: Annotated[list[str], "the keywords to search for in the vector store"],
    ) -> tuple[str, List[Document]]:
    """Retrieve information based on keywords."""
    if not keywords:
        return "No relevant keywords found.", []
    retrieved_docs = vector_store1.similarity_search(keywords, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
# Initialize language model
llm = init_chat_model("llama3-8b-8192", model_provider="groq")


agent_executor = create_react_agent(llm, [check, retrieve])

display(Image(agent_executor.get_graph().draw_mermaid_png()))