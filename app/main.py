import os
from flask import json
import numpy as np
import re
from typing import List, Optional, Dict, Any
import logging
import traceback
import getpass

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
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter LangChain/LangSmith API key (optional - press Enter to skip):")

# LangChain imports
try:
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
    from langchain_groq import ChatGroq
    from pydantic import BaseModel, Field
    from flask import Flask, request, jsonify
    
    # Optional Presidio imports
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        presidio_available = True
    except ImportError:
        presidio_available = False
        print("Warning: Presidio not available, using regex fallback for sanitization")
        
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages using:")
    print("pip install langchain langchain-core langchain-community langchain-huggingface langchain-groq")
    print("pip install pandas numpy flask sentence-transformers transformers torch")
    exit(1)

# === SAMPLE DATA GENERATION ===

def create_sample_rules_data():
    """Create sample rules data"""
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
        "Processing of special categories of personal data is prohibited without explicit consent",
        "Data controllers must demonstrate compliance with data protection principles",
        "Personal data breaches must be reported within 72 hours",
        "Data subjects have the right to erasure (right to be forgotten)",
        "Data subjects have the right to restrict processing of their personal data",
        "Automated decision-making requires human oversight and explanation rights"
    ]
    
    documents = []
    for i, text in enumerate(rules_text):
        doc = Document(
            page_content=text,
            metadata={"source": f"regulation_{i+1}", "section": f"article_{i+1}"}
        )
        documents.append(doc)
    
    return documents

def create_sample_patient_data():
    """Create sample patient data"""
    np.random.seed(42)
    n_records = 100
    
    data = []
    first_names = ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Mary", "James", "Jennifer"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    
    for i in range(n_records):
        record = {
            "Id": f"P{i+1:04d}",
            "BIRTHDATE": f"19{np.random.randint(50, 99)}-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
            "SSN": f"{np.random.randint(100, 999)}-{np.random.randint(10, 99)}-{np.random.randint(1000, 9999)}",
            "FIRST": np.random.choice(first_names),
            "LAST": np.random.choice(last_names),
            "GENDER": np.random.choice(["M", "F"]),
            "RACE": np.random.choice(["white", "black", "asian", "hispanic"]),
            "ETHNICITY": np.random.choice(["nonhispanic", "hispanic"]),
            "ADDRESS": f"{np.random.randint(100, 9999)} {np.random.choice(['Main', 'Oak', 'Park', 'First'])} St",
            "CITY": np.random.choice(["Boston", "New York", "Chicago", "Los Angeles", "Houston"]),
            "STATE": np.random.choice(["MA", "NY", "IL", "CA", "TX"]),
            "ZIP": f"{np.random.randint(10000, 99999)}",
            "HEALTHCARE_EXPENSES": round(np.random.normal(5000, 2000), 2),
            "HEALTHCARE_COVERAGE": np.random.choice(["Medicare", "Medicaid", "Private", "None"])
        }
        data.append(record)
    
    return data

# === GLOBAL INITIALIZATION ===

print("Initializing Privacy-Preserving Multi-Agent System...")

# Initialize LLM models for different agents
try:
    print("Setting up LLM models...")
    # Rules Agent LLM - specialized for rule checking
    rules_llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=2000,
        timeout=60
    )
    
    # Data Agent LLM - specialized for data processing
    data_llm = ChatGroq(
        model="openai/gpt-oss-20b", 
        temperature=0,
        max_tokens=2000,
        timeout=60
    )
    
    # Main Orchestrator LLM - controls the overall workflow
    main_llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=3000,
        timeout=60
    )
    
    print("✓ LLM models initialized")
    
except Exception as e:
    print(f"Error initializing LLM models: {e}")
    print("Please check your GROQ_API_KEY")
    exit(1)

# Initialize embeddings and vector stores
try:
    print("Loading embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Create rules vector store
    vector_store_rules = InMemoryVectorStore(embeddings)
    rules_docs = create_sample_rules_data()
    vector_store_rules.add_documents(rules_docs)
    
    # Sample data for patients
    sample_patients = create_sample_patient_data()
    
    print("✓ Vector store and sample data loaded")
    
except Exception as e:
    print(f"Error initializing embeddings/vector stores: {e}")
    print("This might be due to missing transformers or torch. Try:")
    print("pip install sentence-transformers transformers torch")
    exit(1)

# Initialize Presidio engines (optional)
if presidio_available:
    try:
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        print("✓ Presidio engines initialized")
    except Exception as e:
        print(f"Warning: Presidio initialization failed: {e}")
        analyzer = None
        anonymizer = None
        presidio_available = False
else:
    analyzer = None
    anonymizer = None

# === AGENT CLASSES ===

class RulesAgent:
    """Specialized agent for privacy rule checking and violation detection"""
    
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.name = "RulesAgent"
        
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query for privacy rule violations"""
        try:
            # Step 1: Extract keywords that might indicate violations
            keywords = self._extract_keywords(query)
            
            # Step 2: Retrieve relevant rules
            relevant_rules = self._retrieve_rules(keywords)
            
            # Step 3: Check for violations
            violation_analysis = self._check_violations(query, keywords, relevant_rules)
            
            return {
                "agent": self.name,
                "keywords": keywords,
                "relevant_rules": relevant_rules,
                "violation_analysis": violation_analysis,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in RulesAgent.analyze_query: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "status": "error"
            }
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords indicating potential privacy violations"""
        prompt = f"""You are a privacy compliance expert. Analyze this query for potential privacy violations.

Look for requests involving:
- Personal identifiers (names, SSNs, addresses, phone numbers, emails)
- Sensitive personal data (health records, financial data)
- Unauthorized access or data sharing
- Lack of legitimate purpose for data processing

Query: "{query}"

Extract keywords that indicate potential privacy violations. Return as comma-separated list or "none" if no violations detected.

Keywords:"""
        
        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            keywords_text = response.content.strip().lower()
            
            if keywords_text == 'none' or not keywords_text:
                return []
            
            return [k.strip() for k in keywords_text.split(',') if k.strip()]
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _retrieve_rules(self, keywords: List[str]) -> str:
        """Retrieve relevant privacy rules based on keywords"""
        if not keywords:
            return "No privacy-related keywords found."
        
        try:
            search_query = " ".join(keywords)
            retrieved_docs = self.vector_store.similarity_search(search_query, k=5)
            
            if not retrieved_docs:
                return "No relevant rules found."
            
            rules_text = []
            for doc in retrieved_docs:
                rules_text.append(f"Rule: {doc.page_content}")
            
            return "\n".join(rules_text)
            
        except Exception as e:
            logger.error(f"Error retrieving rules: {e}")
            return "Error retrieving rules."
    
    def _check_violations(self, query: str, keywords: List[str], rules: str) -> Dict[str, Any]:
        """Check if query violates privacy rules"""
        if not keywords or not rules or rules == "No relevant rules found.":
            return {
                "has_violations": False,
                "severity": "none",
                "details": "No violations detected.",
                "violated_rules": [],
                "corrective_action": "none"
            }
        
        prompt = f"""You are a privacy compliance officer. Analyze this query against privacy rules.

Query: "{query}"
Detected keywords: {keywords}

Privacy Rules:
{rules}

Determine:
1. Does this query violate privacy rules? (yes/no)
2. Violation severity: high/medium/low/none
3. What corrective action is needed? (reject/correct/allow)

Respond with:
VIOLATIONS: yes/no
SEVERITY: high/medium/low/none
ACTION: reject/correct/allow
DETAILS: explanation
"""
        
        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            analysis = response.content.strip()
            
            # Parse structured response
            has_violations = "VIOLATIONS: yes" in analysis.lower()
            
            severity = "none"
            if "SEVERITY: high" in analysis.lower():
                severity = "high"
            elif "SEVERITY: medium" in analysis.lower():
                severity = "medium"
            elif "SEVERITY: low" in analysis.lower():
                severity = "low"
            
            corrective_action = "allow"
            if "ACTION: reject" in analysis.lower():
                corrective_action = "reject"
            elif "ACTION: correct" in analysis.lower():
                corrective_action = "correct"
            
            return {
                "has_violations": has_violations,
                "severity": severity,
                "details": analysis,
                "violated_rules": keywords if has_violations else [],
                "corrective_action": corrective_action
            }
        except Exception as e:
            logger.error(f"Error checking violations: {e}")
            return {
                "has_violations": False,
                "severity": "none", 
                "details": f"Error analyzing violations: {str(e)}",
                "violated_rules": [],
                "corrective_action": "allow"
            }

class DataAgent:
    """Specialized agent for data processing, sanitization, and analysis"""
    
    def __init__(self, llm, sample_data):
        self.llm = llm
        self.sample_data = sample_data
        self.name = "DataAgent"
        
    def process_data_request(self, query: str, rules_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process data request based on rules analysis"""
        try:
            # Step 1: Handle violations if any
            violation_analysis = rules_analysis.get("violation_analysis", {})
            if violation_analysis.get("corrective_action") == "reject":
                return {
                    "agent": self.name,
                    "status": "rejected",
                    "reason": "High severity privacy violations",
                    "message": f"Request denied: {violation_analysis.get('details', 'Privacy violations detected')}"
                }
            
            # Step 2: Correct query if needed
            processed_query = query
            if violation_analysis.get("corrective_action") == "correct":
                processed_query = self._correct_query(query, rules_analysis)
            
            # Step 3: Sanitize the query
            sanitized_query = self._sanitize_text(processed_query)
            
            # Step 4: Extract and analyze data
            data_results = self._analyze_data(sanitized_query, query)
            
            return {
                "agent": self.name,
                "status": "completed",
                "original_query": query,
                "processed_query": processed_query,
                "sanitized_query": sanitized_query,
                "data_results": data_results,
                "correction_applied": processed_query != query
            }
            
        except Exception as e:
            logger.error(f"Error in DataAgent.process_data_request: {e}")
            return {
                "agent": self.name,
                "status": "error",
                "error": str(e)
            }
    
    def _correct_query(self, query: str, rules_analysis: Dict[str, Any]) -> str:
        """Correct query to comply with privacy rules"""
        violation_details = rules_analysis.get("violation_analysis", {}).get("details", "")
        
        prompt = f"""You are a privacy compliance specialist. Correct this query to comply with privacy rules.

Original query: "{query}"

Violation analysis: {violation_details}

Rewrite the query to:
1. Remove specific personal identifiers (names, SSNs, addresses)
2. Focus on aggregate/statistical data instead
3. Ensure legitimate purpose for data access

Corrected query:"""
        
        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            corrected = response.content.strip()
            
            # Additional regex sanitization
            corrected = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', corrected)
            corrected = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', corrected)
            
            return corrected
        except Exception as e:
            logger.error(f"Error correcting query: {e}")
            return query
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text by removing PII"""
        try:
            if presidio_available and analyzer and anonymizer:
                results = analyzer.analyze(text=text, entities=[], language="en")
                sanitized = anonymizer.anonymize(text=text, analyzer_results=results).text
            else:
                # Regex fallback
                sanitized = text
                sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', sanitized)
                sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', sanitized)
                sanitized = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', sanitized)
                sanitized = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', sanitized)
            
            return sanitized
        except Exception as e:
            logger.error(f"Error sanitizing text: {e}")
            return text
    
    def _analyze_data(self, sanitized_query: str, original_query: str) -> str:
        """Analyze data based on sanitized query"""
        try:
            # Calculate dataset statistics
            stats = self._calculate_statistics()
            
            # Generate response
            prompt = f"""You are a healthcare data analyst. Provide a response based on this query and dataset.

Original query: "{original_query}"
Sanitized query: "{sanitized_query}"

Dataset Statistics:
{stats}

Provide a clear, helpful response that:
1. Answers the specific question asked
2. Uses only the actual data provided
3. Maintains privacy by not exposing individual records

Response:"""
            
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error analyzing data: {e}")
            return "Unable to analyze data at this time."
    
    def _calculate_statistics(self) -> str:
        """Calculate comprehensive dataset statistics"""
        try:
            total = len(self.sample_data)
            
            # Calculate distributions
            gender_dist = {}
            race_dist = {}
            coverage_dist = {}
            expenses = []
            
            for patient in self.sample_data:
                gender = patient.get('GENDER', 'Unknown')
                gender_dist[gender] = gender_dist.get(gender, 0) + 1
                
                race = patient.get('RACE', 'Unknown')
                race_dist[race] = race_dist.get(race, 0) + 1
                
                coverage = patient.get('HEALTHCARE_COVERAGE', 'Unknown')
                coverage_dist[coverage] = coverage_dist.get(coverage, 0) + 1
                
                expense = patient.get('HEALTHCARE_EXPENSES', 0)
                if expense and expense > 0:
                    expenses.append(expense)
            
            avg_expense = sum(expenses) / len(expenses) if expenses else 0
            
            return f"""Total Patients: {total}
Gender Distribution: {dict(gender_dist)}
Race Distribution: {dict(race_dist)}
Coverage Distribution: {dict(coverage_dist)}
Average Healthcare Expense: ${avg_expense:.2f}
Expense Range: ${min(expenses):.2f} - ${max(expenses):.2f}"""
        
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return "Error calculating dataset statistics"

class MainOrchestrator:
    """Main agent that orchestrates the privacy-preserving workflow"""
    
    def __init__(self, main_llm, rules_agent, data_agent):
        self.llm = main_llm
        self.rules_agent = rules_agent
        self.data_agent = data_agent
        self.name = "MainOrchestrator"
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """Main orchestration logic"""
        try:
            logger.info(f"MainOrchestrator processing: {query}")
            
            # Step 1: Delegate to rules agent
            logger.info("Delegating to RulesAgent...")
            rules_result = self.rules_agent.analyze_query(query)
            
            # Step 2: Make decision based on rules analysis
            decision = self._make_decision(query, rules_result)
            
            if decision["action"] == "reject":
                return {
                    "status": "rejected",
                    "query": query,
                    "reason": decision["reason"],
                    "violations": rules_result.get("violation_analysis", {}),
                    "message": decision["message"]
                }
            
            # Step 3: Delegate to data agent
            logger.info("Delegating to DataAgent...")
            data_result = self.data_agent.process_data_request(query, rules_result)
            
            # Step 4: Generate final response
            final_response = self._generate_final_response(query, rules_result, data_result)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in MainOrchestrator: {e}")
            return {
                "status": "error",
                "query": query,
                "error": str(e),
                "message": "Internal system error occurred"
            }
    
    def _make_decision(self, query: str, rules_result: Dict[str, Any]) -> Dict[str, Any]:
        """Make processing decision based on rules analysis"""
        violation_analysis = rules_result.get("violation_analysis", {})
        
        if violation_analysis.get("corrective_action") == "reject":
            return {
                "action": "reject",
                "reason": "High severity privacy violations detected",
                "message": f"Request denied due to privacy violations: {violation_analysis.get('details', '')}"
            }
        
        return {
            "action": "proceed",
            "reason": "Query approved for processing",
            "message": "Processing query with appropriate privacy protections"
        }
    
    def _generate_final_response(self, query: str, rules_result: Dict[str, Any], 
                               data_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final response"""
        
        violation_analysis = rules_result.get("violation_analysis", {})
        has_violations = violation_analysis.get("has_violations", False)
        
        # Prepare violation warning if needed
        violation_message = ""
        if has_violations and violation_analysis.get("corrective_action") == "correct":
            violation_message = f"""⚠️ PRIVACY NOTICE: Your request was modified for compliance.

Detected issues: {violation_analysis.get('details', 'Privacy violations')}
Relevant rules: {rules_result.get('relevant_rules', 'Data protection regulations')}

Modified query: "{data_result.get('processed_query', query)}"

"""
        
        # Get data results
        data_response = data_result.get("data_results", "No data results available")
        
        # Combine messages
        if violation_message:
            final_output = violation_message + "\n📊 ANALYSIS RESULTS:\n" + data_response
        else:
            final_output = data_response
        
        return {
            "status": "success",
            "query": query,
            "output": final_output,
            "violations": {
                "found": has_violations,
                "severity": violation_analysis.get("severity", "none"),
                "details": violation_analysis.get("details") if has_violations else None,
                "corrective_action": violation_analysis.get("corrective_action", "none")
            },
            "processing_info": {
                "rules_analysis": rules_result,
                "data_processing": {
                    "query_corrected": data_result.get("correction_applied", False),
                    "sanitization_applied": True,
                    "data_extracted": True
                }
            }
        }

# === MAIN SYSTEM ===

class PrivacyPreservingMultiAgentSystem:
    """Complete multi-agent privacy-preserving system"""
    
    def __init__(self):
        # Initialize individual agents
        self.rules_agent = RulesAgent(rules_llm, vector_store_rules)
        self.data_agent = DataAgent(data_llm, sample_patients)
        self.main_orchestrator = MainOrchestrator(main_llm, self.rules_agent, self.data_agent)
        
        # Cache for performance
        self.query_cache = {}
        
        print("✓ Multi-agent system initialized")
        print(f"  - Rules Agent: {self.rules_agent.name}")
        print(f"  - Data Agent: {self.data_agent.name}")  
        print(f"  - Main Orchestrator: {self.main_orchestrator.name}")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query through multi-agent system"""
        global RESULT
        
        # Check cache
        if query in self.query_cache:
            logger.info("Returning cached response")
            return self.query_cache[query]
        
        try:
            # Delegate to main orchestrator
            result = self.main_orchestrator.process_query(query)
            
            # Cache result
            self.query_cache[query] = result
            RESULT = result
            
            return result
            
        except Exception as e:
            logger.error(f"System error: {e}")
            traceback.print_exc()
            
            error_result = {
                "status": "system_error",
                "query": query,
                "error": str(e),
                "message": "System encountered an unexpected error"
            }
            
            return error_result

# === FLASK API ===

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "message": "Privacy-Preserving Multi-Agent System API is running",
        "agents": {
            "rules_agent": "RulesAgent",
            "data_agent": "DataAgent",
            "main_orchestrator": "MainOrchestrator"
        },
        "presidio_available": presidio_available,
        "sample_patients": len(sample_patients)
    })

@app.route("/api", methods=["POST"])
def query_endpoint():
    """Main API endpoint for processing queries"""
    try:
        # Parse JSON data
        data = request.get_json(force=True)
        logger.info(f"API request received: {data}")
        
        if not data or "query" not in data:
            return jsonify({"status": "error", "message": "No query provided"}), 400
        
        # Process through multi-agent system
        result = system.process_query(data["query"])
        
        # Return appropriate response based on status
        if result["status"] == "success":
            return jsonify({
                "output": result["output"], 
                "violations": result.get("violations", {}),
                "processing_info": result.get("processing_info", {})
            }), 200
            
        elif result["status"] == "rejected":
            return jsonify({
                "status": "rejected",
                "message": result.get("message", "Request rejected due to privacy violations"),
                "violations": result.get("violations", {}),
                "reason": result.get("reason", "Privacy rule violations")
            }), 403
            
        else:  # error or system_error
            return jsonify({
                "status": "error",
                "message": result.get("message", "Processing error"),
                "error": result.get("error", "Unknown error")
            }), 500
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"API Error: {e}\n{error_trace}")
        return jsonify({
            "status": "error", 
            "message": str(e), 
            "traceback": error_trace
        }), 500

# === MAIN EXECUTION ===

if __name__ == "__main__":
    import sys
    
    # Initialize the system
    print("Creating multi-agent system...")
    system = PrivacyPreservingMultiAgentSystem()
    print("✓ System ready!")

    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Run Flask API
        print("\n🚀 Starting Privacy-Preserving Multi-Agent System API...")
        print("📡 Endpoints:")
        print("  GET  / - Health check")
        print("  POST /api - Query processing")
        print(f"🔒 Privacy protection: {'Enhanced (Presidio)' if presidio_available else 'Standard (Regex)'}")
        print("🌐 Server starting on http://localhost:5000")
        print("=" * 50)
        
        app.run(debug=True, port=5000, host='0.0.0.0')
        
    else:
        # Test mode - run sample queries
        print("\n🧪 Running in test mode...")
        print("=" * 50)
        
        test_queries = [
            "How many patients do we have?",
            "Show me statistics about patient demographics", 
            "Find John Smith with SSN 123-45-6789",
            "What's the average healthcare expense?"
        ]
        
        for i, test_query in enumerate(test_queries, 1):
            print(f"\n--- Test {i}: {test_query} ---")
            try:
                result = system.process_query(test_query)
                print(f"Status: {result['status']}")
                print(f"Output: {result['output'][:200]}...")
                if result.get('violations', {}).get('found'):
                    print(f"Violations: {result['violations']['severity']}")
            except Exception as e:
                print(f"Error: {e}")
        
        print(f"\n🎯 Final result stored in RESULT: {RESULT.get('status', 'none')}")
