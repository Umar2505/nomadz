import os
from flask import json
import numpy as np
import re
from typing import List, Optional, Annotated
import logging
import traceback
import getpass
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_groq import ChatGroq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
RESULT = {}

# Set environment variables
os.environ["LANGSMITH_TRACING"] = "true"

# Prompt for API keys if not set
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq:")

if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass(
        "Enter LangChain/LangSmith API key:"
    )

# LangChain imports
try:
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.tools import tool
    from langchain_core.documents import Document
    from langchain.chat_models import init_chat_model
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langgraph.prebuilt import create_react_agent
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from pydantic import BaseModel, Field
    from flask import Flask, request, jsonify
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages using:")
    print(
        "pip install langchain langgraph langchain-core langchain-community langchain-huggingface"
    )
    print("pip install presidio-analyzer presidio-anonymizer pandas numpy PyPDF2 flask")
    print("pip install sentence-transformers transformers torch")
    exit(1)

# === SAMPLE DATA GENERATION (replaces hardcoded file paths) ===


def create_sample_rules_data():
    """Create sample rules data to replace PDF loading"""
    rules_text = [
        "Personal data must be processed lawfully, fairly and in a transparent manner",
        "Personal data shall be collected for specified, explicit and legitimate purposes",
        "Personal data shall be adequate, relevant and limited to what is necessary",
        "Personal data shall be accurate and, where necessary, kept up to date",
        "Personal data shall be kept in a form which permits identification for no longer than necessary",
        "Personal data shall be processed in a manner that ensures appropriate security",
        "Organizations must implement appropriate technical and organizational measures",
        "Data subjects have the right to be informed about data processing",
        "Data subjects have the right to access their personal data",
        "Data subjects have the right to rectification of inaccurate data",
        "Sensitive personal data requires explicit consent or other lawful basis",
        "Cross-border data transfers require adequate protection measures",
    ]

    # Create documents
    documents = []
    for i, text in enumerate(rules_text):
        doc = Document(
            page_content=text,
            metadata={"source": f"regulation_{i+1}", "section": f"article_{i+1}"},
        )
        documents.append(doc)

    return documents


def create_sample_patient_data():
    """Create sample patient data"""
    np.random.seed(42)
    n_records = 1000

    data = []
    for i in range(n_records):
        record = {
            "Id": f"P{i+1:04d}",
            "BIRTHDATE": f"19{np.random.randint(50, 99)}-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
            "SSN": f"{np.random.randint(100, 999)}-{np.random.randint(10, 99)}-{np.random.randint(1000, 9999)}",
            "FIRST": np.random.choice(
                ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Mary"]
            ),
            "LAST": np.random.choice(
                [
                    "Smith",
                    "Johnson",
                    "Williams",
                    "Brown",
                    "Jones",
                    "Garcia",
                    "Miller",
                    "Davis",
                ]
            ),
            "GENDER": np.random.choice(["M", "F"]),
            "RACE": np.random.choice(["white", "black", "asian", "hispanic"]),
            "ETHNICITY": np.random.choice(["nonhispanic", "hispanic"]),
            "ADDRESS": f"{np.random.randint(100, 9999)} {np.random.choice(['Main', 'Oak', 'Park', 'First'])} St",
            "CITY": np.random.choice(
                ["Boston", "New York", "Chicago", "Los Angeles", "Houston"]
            ),
            "STATE": np.random.choice(["MA", "NY", "IL", "CA", "TX"]),
            "ZIP": f"{np.random.randint(10000, 99999)}",
            "HEALTHCARE_EXPENSES": round(np.random.normal(5000, 2000), 2),
            "HEALTHCARE_COVERAGE": np.random.choice(
                ["Medicare", "Medicaid", "Private", "None"]
            ),
        }
        data.append(record)

    return data


# === PYDANTIC MODELS ===


class Person(BaseModel):
    """Information about a person."""

    id: Optional[str] = Field(
        default=None, description="Unique identifier for the person"
    )
    birthdate: Optional[str] = Field(
        default=None, description="Date of birth (YYYY-MM-DD)"
    )
    ssn: Optional[str] = Field(default=None, description="Social Security Number")
    first: Optional[str] = Field(default=None, description="First name")
    last: Optional[str] = Field(default=None, description="Last name")
    gender: Optional[str] = Field(default=None, description="Gender")
    race: Optional[str] = Field(default=None, description="Race")
    ethnicity: Optional[str] = Field(default=None, description="Ethnicity")
    address: Optional[str] = Field(default=None, description="Street address")
    city: Optional[str] = Field(default=None, description="City")
    state: Optional[str] = Field(default=None, description="State")
    zip: Optional[str] = Field(default=None, description="ZIP code")
    healthcare_expenses: Optional[float] = Field(
        default=None, description="Healthcare expenses"
    )
    healthcare_coverage: Optional[str] = Field(
        default=None, description="Healthcare coverage type"
    )


class CorrectedOutput(BaseModel):
    """Structured output after applying violated rule corrections."""

    corrected_text: str = Field(
        description="The corrected text based on violated rules"
    )
    applied_rules: List[str] = Field(
        description="List of rules that were applied for correction"
    )


# === GLOBAL INITIALIZATION ===

# Initialize LLM models
try:
    llm_tool = init_chat_model("deepseek-r1-distill-llama-70b", model_provider="groq")
    llm_ruler = init_chat_model("deepseek-r1-distill-llama-70b", model_provider="groq")
    llm_main = init_chat_model("deepseek-r1-distill-llama-70b", model_provider="groq")
except Exception as e:
    print(f"Error initializing LLM models: {e}")
    print("Please check your GROQ_API_KEY")
    exit(1)

# Initialize embeddings and vector stores
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Create rules vector store
    vector_store_rules = InMemoryVectorStore(embeddings)
    rules_docs = create_sample_rules_data()
    vector_store_rules.add_documents(rules_docs)

    # Sample data for patients
    sample_patients = create_sample_patient_data()

except Exception as e:
    print(f"Error initializing embeddings/vector stores: {e}")
    exit(1)

# Initialize Presidio engines
try:
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
except Exception as e:
    print(f"Error initializing Presidio: {e}")
    print("Note: Presidio may require additional setup for full functionality")
    analyzer = None
    anonymizer = None

# === TOOLS DEFINITIONS ===

# Prompts
template = """You are an AI assistant that finds keywords from the query parts that can cause rule violations to retrieve relevant rules and regulations from another tool. You should return a list of keywords that are relevant to the query separated by commas. If there are no relevant keywords, return an empty list.

Query: {query}

Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)

prompt_for_data = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)


@tool
def parser(input: Annotated[str, "the query to analyze for rule keywords"]) -> str:
    """Analyze the query for possible rule violation keywords."""
    messages = custom_rag_prompt.invoke({"query": input})
    response = llm_tool.invoke(messages)
    if hasattr(response, "content"):
        keywords_str = response.content
    else:
        keywords_str = str(response)

    # Parse keywords from response
    if keywords_str and keywords_str.strip():
        keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
    else:
        keywords = []

    return json.dumps(keywords) if keywords else "No keywords found"


@tool
def retriever_rules(
    keywords: Annotated[str, "keywords to search for in vector store"],
) -> str:
    """Retrieve information based on keywords."""
    try:
        if not keywords or len(keywords) == 0:
            return "No relevant keywords found."

        # Join keywords for search
        retrieved_docs = vector_store_rules.similarity_search(keywords, k=3)

        if not retrieved_docs:
            return "No relevant rules found."

        serialized = "\\n\\n".join(
            [
                f"Source: {doc.metadata}\\nContent: {doc.page_content}"
                for doc in retrieved_docs
            ]
        )

        return serialized
    except Exception as e:
        logger.error(f"Error in retriever_rules tool: {e}")
        return "Error retrieving rules."


@tool
def violation_corrector(
    input: Annotated[str, "The input text to analyze"],
    violated_rules: Annotated[List[str], "List of rules violated by the input"],
) -> CorrectedOutput:
    """Corrects the input text according to violated rules."""
    try:
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
        # return result
        # ensure result is a dict or CorrectedOutput instance converted to dict
        if isinstance(result, CorrectedOutput):
            return result.model_dump()
        # or if result is an object with attributes:
        return {
            "corrected_text": getattr(result, "corrected_text", ""),
            "applied_rules": getattr(result, "applied_rules", []),
        }

    except Exception as e:
        logger.error(f"Error in violation_corrector: {e}")
        return CorrectedOutput(corrected_text=input, applied_rules=[])


class SanitizerInput(BaseModel):
    corrected_text: str
    applied_rules: list[str] = []


@tool
def sanitizer(input: SanitizerInput) -> dict:
    """General sanitizer that uses Microsoft Presidio to anonymize sensitive data."""
    try:
        text = input.corrected_text

        if analyzer is None or anonymizer is None:
            sanitized_text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "XXX-XX-XXXX", text)
            sanitized_text = re.sub(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "[EMAIL]",
                sanitized_text,
            )
        else:
            results = analyzer.analyze(text=text, entities=[], language="en")
            sanitized_text = anonymizer.anonymize(
                text=text, analyzer_results=results
            ).text

        return {"corrected_text": sanitized_text, "applied_rules": input.applied_rules}
    except Exception as e:
        logger.error(f"Error in sanitizer: {e}")
        return {
            "corrected_text": input.corrected_text,
            "applied_rules": input.applied_rules,
        }


@tool
def retriever_data(
    input: Annotated[CorrectedOutput, "The sanitized corrected text and applied rules"],
):
    """Retrieve structured Person information based on sanitized input text."""
    try:
        structured_llm = llm_tool.with_structured_output(schema=Person)

        # Use the sanitized corrected text from CorrectedOutput
        prompt = prompt_for_data.invoke({"text": input.corrected_text})
        result = structured_llm.invoke(prompt)

        return " ,".join(result)
    except Exception as e:
        logger.error(f"Error in retriever_data: {e}")
        return Person()


@tool
def average(numbers: List[float]) -> float:
    """Calculate the average of a list of numbers."""
    try:
        if not numbers:
            return 0.0
        return sum(numbers) / len(numbers)
    except Exception as e:
        logger.error(f"Error in average: {e}")
        return 0.0


@tool
def total(numbers: List[float]) -> float:
    """Calculate the total sum of a list of numbers."""
    try:
        if not numbers:
            return 0.0
        return sum(numbers)
    except Exception as e:
        logger.error(f"Error in total: {e}")
        return 0.0


@tool
def greater_than(a: float, b: float) -> bool:
    """Return True if a is greater than b, otherwise False."""
    try:
        return a > b
    except Exception:
        return False


@tool
def less_than(a: float, b: float) -> bool:
    """Return True if a is less than b, otherwise False."""
    try:
        return a < b
    except Exception:
        return False


# === AGENT CREATION ===

# Create rules agent
try:
    rules_agent_executor = create_react_agent(llm_ruler, [parser, retriever_rules])
except Exception as e:
    logger.error(f"Error creating rules agent: {e}")
    rules_agent_executor = None

# Create data agent
try:
    data_agent_tools = [
        violation_corrector,
        retriever_data,
        sanitizer,
        average,
        total,
        less_than,
        greater_than,
    ]
    data_agent_executor = create_react_agent(llm_tool, data_agent_tools)
except Exception as e:
    logger.error(f"Error creating data agent: {e}")
    data_agent_executor = None

# === MAIN AGENT TOOLS ===


@tool
def rules_agent(
    query: Annotated[str, "the user query to analyze for rule violations"],
) -> str:
    """Detect possible rule violations based on user input."""
    try:
        if rules_agent_executor is None:
            return ""

        template_rules_agent = """You are an AI assistant that detects possible rule violations based on user input.
                                You have access to two tools:
                                1) parser(input: str) → list[str] - analyzes input and returns keywords
                                2) retriever_rules(keywords: list[str]) → str - retrieves rule violations

                                Behavior:
                                - First call parser() with the input string
                                - If parser() returns keywords, pass them to retriever_rules()
                                - Return the list of possible rule violations retrieved

                                Query: {query}

                                Answer:"""

        rag_prompt_rules = PromptTemplate.from_template(template_rules_agent)
        messages = rag_prompt_rules.invoke({"query": query})

        result = rules_agent_executor.invoke(
            {"messages": [{"role": "user", "content": messages.text}]}
        )

        # Extract violations from the result
        if "messages" in result and len(result["messages"]) > 0:
            last_message = result["messages"][-1]
            content = getattr(last_message, "content", None)
            if content is None or (
                isinstance(content, (list, dict)) and len(content) == 0
            ):
                # Handle or convert content to string here
                content = str(content) if content else ""
            # Use this safe content string

            # return ""
            return content

    except Exception as e:
        logger.error(f"Error in rules_agent: {e}")
    return ""


@tool
def data_agent(query: Annotated[str, "the user query to find data"]) -> str:
    """Analyze, correct, and sanitize user input while ensuring compliance with rules."""
    try:
        if data_agent_executor is None:
            return ""

        template_violation_correction_agent = """You are an AI assistant responsible for analyzing, correcting, 
                                                and sanitizing user input while ensuring compliance with rules.
                                                You have access to tools for:
                                                1) violation_corrector - corrects text based on violated rules
                                                2) sanitizer - sanitizes corrected text to remove PII
                                                3) retriever_data - retrieves structured information

                                                Behavior:
                                                1. Receive user input and potentially violated rules
                                                2. Use violation_corrector to generate corrected text
                                                3. Use sanitizer to remove sensitive data
                                                4. Use retriever_data to extract structured information

                                                User Query: {query}

                                                Final Output:"""

        rag_prompt_data = PromptTemplate.from_template(
            template_violation_correction_agent
        )
        messages = rag_prompt_data.invoke({"query": query})

        result = data_agent_executor.invoke(
            {"messages": [{"role": "user", "content": messages.text}]}
        )

        # Extract result from the response
        if "messages" in result and len(result["messages"]) > 0:
            last_message = result["messages"][-1]
            content = getattr(last_message, "content", None)
            if content is None or (
                isinstance(content, (list, dict)) and len(content) == 0
            ):
                # Handle or convert content to string here
                content = str(content) if content else ""
            # Use this safe content string
            return content
        return ""
    except Exception as e:
        logger.error(f"Error in data_agent: {e}")
        return ""


# Create main agent
try:
    main_agent_executor = create_react_agent(llm_main, [rules_agent, data_agent])
except Exception as e:
    logger.error(f"Error creating main agent: {e}")
    main_agent_executor = None


class PrivacyPreservingAgent:
    """Privacy-Preserving Agent with caching and AI-driven orchestration."""

    def __init__(self):
        # Cache: query → response
        self.query_cache: dict[str, dict] = {}

        # === LLM ===
        self.llm = ChatGroq(
            model="deepseek-r1-distill-llama-70b",
            temperature=0,
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
            # other params...
        )

        # === Register tools ===
        self.tools = [
            Tool.from_function(
                func=rules_agent.invoke,
                name="rules_agent",
                description="Check for rule violations in the query.",
            ),
            Tool.from_function(
                func=violation_corrector.invoke,
                name="violation_corrector",
                description="Correct text based on violations.",
            ),
            Tool.from_function(
                func=sanitizer.invoke,
                name="sanitizer",
                description="Sanitize text by removing sensitive information.",
            ),
            Tool.from_function(
                func=retriever_data.invoke,
                name="retriever_data",
                description="Extract structured data from sanitized text.",
            ),
        ]

        # === Create LangChain agent ===
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={
                "system_message": (
                    "You are a privacy-preserving AI assistant. Your task is to process user queries securely "
                    "and provide accurate answers from the dataset. Always follow these steps in order:\n\n"
                    "1. Check for rule violations using `rules_agent`. Record any violations.\n"
                    "2. Correct the input using `violation_corrector` if violations are found.\n"
                    "3. Sanitize the text using `sanitizer` to remove any sensitive information.\n"
                    "4. Extract structured data using `retriever_data`.\n"
                    "5. Compute results deterministically from the structured data (e.g., demographics, counts, most popular ethnicity).\n"
                    "6. Generate a final, human-readable answer that includes:\n"
                    "   - Your reasoning steps (Thought)\n"
                    "   - Any violations found\n"
                    "   - Corrected and sanitized text\n"
                    "   - The final answer from the dataset\n\n"
                    "Important rules:\n"
                    "- Never skip any step.\n"
                    "- Do not invent data — always derive answers from the structured dataset.\n"
                    "- Include violations in your reasoning if any.\n"
                    "- Use sanitized data only when generating the final answer.\n"
                    "- Be concise, clear, and factual in the Final Answer."
                    "Response format:\n\n"
                    "Thought: <step-by-step reasoning including any rule violations>\n"
                    "Action: <tool used at this step>\n"
                    "Action Input: <JSON input to the tool>\n\n"
                    "...repeat for all steps in your reasoning process...\n\n"
                    "Final Answer: <concise, clear, user-facing answer derived from structured data>\n\n"
                )
            },
        )

        logger.info("Privacy-Preserving Agent initialized")

    def process_query(self, query: str) -> dict:
        """Run the full privacy-preserving pipeline as an agent."""
        global RESULT
        # === 0. Cache lookup ===
        if query in self.query_cache:
            logger.info("Returning cached response for query")
            return self.query_cache[query]

        try:
            # === Let the agent decide tool usage ===
            logger.info("Agent orchestrating workflow...")
            answer = self.agent.run(query)

            response = {
                "status": "success",
                "query": query,
                "output": answer,
            }
            RESULT = response
            # === Cache ===
            self.query_cache[query] = response

        except Exception as e:
            logger.error(f"Error in process_query: {e}")
            response = {"status": "error", "query": query, "output": str(e)}

        return response


# === FLASK API ===

# Initialize Flask app
app = Flask(__name__)
system = PrivacyPreservingAgent()


@app.route("/api", methods=["POST"])
def query():
    """Main API endpoint for processing queries"""
    try:
        # Parse JSON data
        data = request.get_json(force=True)
        print("Data received:", data)

        if not data or "query" not in data:
            return jsonify({"status": "error", "message": "No query provided"}), 400

        # Process the query
        result = system.process_query(data["query"])

        return jsonify({"output": result["output"], "violations": []}), 200

    except Exception as e:
        error_trace = traceback.format_exc()
        print("ERROR TRACEBACK:\\n", error_trace)
        return (
            jsonify({"status": "error", "message": str(e), "traceback": error_trace}),
            500,
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Run Flask API
        print("Starting Privacy-Preserving Multi-Agent System API...")
        print("Access endpoints:")
        print("  POST /api - Main query processing")
        app.run(debug=True, port=5000)
        print("The result is:", RESULT)
