from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from typing_extensions import Annotated, List
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

llm_tool = init_chat_model("llama3-8b-8192", model_provider="groq")

file_path1 = "app/data/CELEX.pdf"
loader1 = PyPDFLoader(file_path1)

rule = loader1.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
rules = text_splitter.split_documents(rule)

# Load and split CSV document
file_path2 = "app/docs/patients.csv"
loader2 = CSVLoader(file_path2, metadata_columns=["Id", "BIRTHDATE", "DEATHDATE", "SSN", "DRIVERS", "PASSPORT", "PREFIX", "FIRST", "LAST", "SUFFIX", "MAIDEN", "MARITAL", "RACE", "ETHNICITY", "GENDER", "BIRTHPLACE", "ADDRESS", "CITY", "STATE", "COUNTY", "ZIP", "LAT", "LON", "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE"])
data = loader2.load()


# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create vector store and add rules
vector_store1 = InMemoryVectorStore(embeddings)
rul = vector_store1.add_documents(rules)

from typing import Optional
from pydantic import BaseModel, Field


class Person(BaseModel):
    """Information about a person."""

    id: Optional[str] = Field(default=None, description="Unique identifier for the person")
    birthdate: Optional[str] = Field(default=None, description="Date of birth of the person (YYYY-MM-DD)")
    deathdate: Optional[str] = Field(default=None, description="Date of death if applicable (YYYY-MM-DD)")
    ssn: Optional[str] = Field(default=None, description="Social Security Number of the person")
    drivers: Optional[str] = Field(default=None, description="Driver's license number of the person")
    passport: Optional[str] = Field(default=None, description="Passport number of the person")
    prefix: Optional[str] = Field(default=None, description="Name prefix (e.g., Mr., Mrs., Dr.)")
    first: Optional[str] = Field(default=None, description="First name of the person")
    last: Optional[str] = Field(default=None, description="Last name (surname/family name) of the person")
    suffix: Optional[str] = Field(default=None, description="Name suffix (e.g., Jr., Sr., III)")
    maiden: Optional[str] = Field(default=None, description="Maiden name of the person, if applicable")
    marital: Optional[str] = Field(default=None, description="Marital status (e.g., single, married, divorced)")
    race: Optional[str] = Field(default=None, description="Race of the person")
    ethnicity: Optional[str] = Field(default=None, description="Ethnicity of the person")
    gender: Optional[str] = Field(default=None, description="Gender of the person")
    birthplace: Optional[str] = Field(default=None, description="Place where the person was born")
    address: Optional[str] = Field(default=None, description="Street address of the person")
    city: Optional[str] = Field(default=None, description="City of residence")
    state: Optional[str] = Field(default=None, description="State of residence")
    county: Optional[str] = Field(default=None, description="County of residence")
    zip: Optional[str] = Field(default=None, description="ZIP or postal code")
    lat: Optional[float] = Field(default=None, description="Latitude coordinate of residence")
    lon: Optional[float] = Field(default=None, description="Longitude coordinate of residence")
    healthcare_expenses: Optional[float] = Field(default=None, description="Total healthcare expenses of the person")
    healthcare_coverage: Optional[str] = Field(default=None, description="Type of healthcare coverage held by the person")

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt_for_data = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)

template = """You are an AI assistant that finds keywords from the query parts that can cause rule violations to retrieve relevant rules and regulations from another tool. You should return a list of keywords that are relevant to the query separated by commas. If there are no relevant keywords, return an empty list.

Query: {query}
Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

@tool
def parser(
    input: Annotated[str, "the query to analyze for rule keywords"],
    ) -> list[str]:
    """Analyze the query for possible rule violation keywords."""
    messages = custom_rag_prompt.invoke({"query": input})
    try:
        keywords = ",".split(llm_tool.invoke(messages))
    except Exception as e:
        keywords = []
    return keywords

@tool(response_format="content_and_artifact")
def retriever_rules(keywords: Annotated[list[str], "the keywords to search for in the vector store"],
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

class CorrectedOutput(BaseModel):
    """Structured output after applying violated rule corrections."""

    corrected_text: str = Field(
        description="The corrected text based on the violated rules and input"
    )
    applied_rules: List[str] = Field(
        description="List of violated rules that were applied for correction"
    )


@tool(response_format="content_and_artifact")
def violation_corrector(
    input: Annotated[str, "The input text to analyze"],
    violated_rules: Annotated[List[str], "List of rules violated by the input"],
) -> CorrectedOutput:
    """
    Corrects the input text according to violated rules and returns a structured response.
    """

    # Create a structured LLM with output schema
    structured_llm = llm_tool.with_structured_output(schema=CorrectedOutput)

    # Build a prompt that asks the model to apply the rules
    prompt = f"""
    You are an AI assistant that corrects input text based on violated rules.

    Input text:
    {input}

    Violated rules:
    {violated_rules}

    Task:
    - Identify the parts of the input text that violate the rules.
    - Correct them while preserving the rest of the input.
    - Return the corrected text as 'corrected_text'.
    - Also return the list of applied rules as 'applied_rules'.
    """

    result = structured_llm.invoke(prompt)
    return result

class CorrectedOutput(BaseModel):
    """Structured output after applying violated rule corrections."""

    corrected_text: str = Field(
        description="The corrected text based on the violated rules and input"
    )
    applied_rules: List[str] = Field(
        description="List of violated rules that were applied for correction"
    )


# Initialize Presidio engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


@tool(response_format="content_and_artifact")
def sanitizer(input: CorrectedOutput) -> CorrectedOutput:
    """
    General sanitizer that uses Microsoft Presidio to anonymize sensitive data
    in the corrected_text field of CorrectedOutput.
    """

    # Analyze the text for PII entities
    results = analyzer.analyze(
        text=input.corrected_text,
        entities=[],  # Empty means "detect all supported entities"
        language="en"
    )

    # Anonymize detected entities
    sanitized_text = anonymizer.anonymize(
        text=input.corrected_text,
        analyzer_results=results
    ).text

    # Return the same schema, but with sanitized text
    return CorrectedOutput(
        corrected_text=sanitized_text,
        applied_rules=input.applied_rules
    )


@tool(response_format="content_and_artifact")
def retriever_data(
    input: Annotated[CorrectedOutput, "The sanitized corrected text and applied rules"],
):
    """Retrieve structured Person information based on sanitized input text."""

    structured_llm = llm_tool.with_structured_output(schema=Person)

    # Use the sanitized corrected text from CorrectedOutput
    prompt = prompt_for_data.invoke({"text": input.corrected_text})

    result = structured_llm.invoke(prompt)

    return result


def guardrails():
    pass