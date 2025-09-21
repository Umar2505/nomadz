# Nomadz
## Overview
This project implements an LLM-based agentic system capable of answering natural language queries on sensitive datasets without exposing private information. It takes a user’s query, checks it against rules, sanitizes it, and then produces a safe, meaningful response.
## Project diagram
<img width="1775" height="1329" alt="Untitled Diagram" src="https://github.com/user-attachments/assets/6c2e9db9-b970-460d-8514-f3efe7ec6d53" />


## Tech Stack

### ⚙️ Backend
- **Python**: Chosen as the language due to its rich ecosystem for machine learning, NLP, and web services.
- **LangChain**: Acts as the LLM orchestration framework for building the agentic system.
- **Flask**: Lightweight Python web framework used to expose the backend as a RESTful API.
- **Groq API**: Provides low-latency, high-throughput inference for LLM queries. Unlike standard cloud LLM providers, Groq specializes in deterministic, hardware-accelerated inference.

### 🖥️ Frontend

- **Framework:** Next.js 15 with React 19, using the App Router for modern UI patterns.  
- **Language:** TypeScript for type safety and better development experience.  
- **Styling:** Tailwind CSS 4 with the new `@import "tailwindcss"` pattern, powered by PostCSS tooling.  
- **Firebase Integration:**  
  - **Authentication** handled with Firebase Auth.  
  - **Data persistence** managed via Cloud Firestore.  
  - Reusable Firebase initialization module + Firestore hooks power the chat workspace page.  

---

### ⚙️ Backend of the Frontend

- A **Next.js Route Handler** (`app/api/chat/route.ts`) acts as a proxy between the client and the upstream Nomadz API.  
- Responsibilities:  
  - Parse JSON requests.  
  - Map errors into consistent responses.  
  - Select the correct API base URL depending on the environment.  
  - Return results as standardized `NextResponse` objects.  


#### 🔧 To get the project working setup Firebase

Follow these steps to enable Firebase in this project:

---

#### 1. Create a Firebase Project
- Go to the [Firebase Console](https://console.firebase.google.com/).  
- Sign in with your Google account.  
- Click **Add Project** and create a new project.  

---

#### 2. Enable Authentication & Firestore
- In the Firebase Console, go to **Build → Authentication** and set up a new authentication method (e.g., Email/Password).  
- Still under **Build**, go to **Firestore Database** and create a new database.  

---

#### 3. Get Your Firebase Config
- In the Firebase Console, click the **⚙️ Settings icon** (Project Settings) in the left sidebar.  
- Under **Your Apps**, create a new **Web App**.  
- Copy the config values that Firebase gives you.  

---

#### 4. Add Environment Variables
- In your project root, create a file called `.env.local`.  
- Paste in the Firebase config values you copied earlier, like this:

```env
NEXT_PUBLIC_FIREBASE_API_KEY=your_api_key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_project.appspot.com
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
NEXT_PUBLIC_FIREBASE_APP_ID=your_app_id
NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID=your_measurement_id
```

#### 5. Set Firestore Security Rules

- In the Firebase Console, go to Firestore Database → Rules.

- Replace the default rules with the following:

  ```rules
  rules_version = '2';
  service cloud.firestore {
    match /databases/{database}/documents {
      
      // Each authenticated user can only access their own user document
      match /users/{userId} {
        allow read, write: if request.auth != null && request.auth.uid == userId;
  
        // Chats inside each user document – only accessible by the same user
        match /chats/{chatId} {
          allow read, write: if request.auth != null && request.auth.uid == userId;
        }
      }
    }
  }
  ```
## Explanation of the Backend System

At a high level, the backend is a multi-agent privacy-preserving system that takes a user’s query, checks it against rules, sanitizes it, and then produces a safe, meaningful response. Here’s how the main pieces work together:

### 1. Environment & Initialization
- The app loads API keys (GROQ_API_KEY, LANGCHAIN_API_KEY) for Groq-powered LLMs and LangChain tracing.
- Sets up logging for debugging.
- Initializes external libraries:
- LangChain for LLM orchestration.
- Presidio for detecting and anonymizing sensitive data (PII).
- Vector stores + embeddings (HuggingFace) to store and search documents like privacy rules.

### 2. Sample Data
- **Rules data**: A list of privacy regulations (e.g., "Personal data must be processed lawfully…") stored in a vector database so the system can retrieve relevant rules.
- **Patient data**: Randomly generated synthetic patient records with identifiers, healthcare info, etc. (used for testing privacy handling).

### 3. Data Models (Pydantic)
- **Person**: Defines the structured fields of patient data (e.g., birthdate, SSN, expenses).
- **CorrectedOutput**: Wraps corrected/sanitized text along with which rules were applied.

These models enforce consistency when LLMs return structured outputs.

### 4. LLMs & Vector Stores
- Three Groq LLMs (llm_tool, llm_ruler, llm_main) are initialized for different agent roles.
- Embeddings (sentence-transformers/all-mpnet-base-v2) let the system perform semantic search on rule texts.

### 5. Privacy Tools
The system defines several "tools" (functions) that agents can call. These are the building blocks of the workflow:

- 🔎 **Parser** → Extracts keywords from queries that might map to rule violations.
- 📚 **Retriever (rules)** → Finds relevant regulations based on those keywords.
- 🛠️ **Violation Corrector** → Uses LLM to fix text that violates rules.
- 🧼 **Sanitizer** → Removes PII (via Presidio or regex fallbacks).
- 🧑‍⚕️ **Retriever (data)** → Extracts structured Person info from corrected/sanitized input.
- ➕ **Math tools** → Helpers like average, total, greater_than, less_than for insights.

### 6. Agents
- **Rules Agent**:
  - Uses parser + retriever_rules.
  - Detects which privacy rules might be violated by the query.
- **Data Agent**:
  - Uses violation_corrector, sanitizer, retriever_data, and math tools.
  - Cleans and anonymizes the data, then extracts safe insights.
- **Main Agent**:
  - Coordinates both specialized agents.
  - First checks rules with the Rules Agent, then applies the Data Agent if needed.
  - Produces the final safe answer to return to the user.

### 7. Query Processing Workflow
Here’s the life of a query step by step:

- User sends query → /api endpoint (Flask).
- Cache check → If the query was processed before, return cached answer.
- Rules Agent → Identifies potential rule violations.
- Data Agent → Corrects violations, sanitizes PII, extracts structured insights.
- Main Agent → Wraps the response in natural language.
- Flask API returns → A safe, privacy-compliant answer with metadata.

## 8. Flask API Layer
- Defines /api POST endpoint.
- Handles request parsing, passes the query into the PrivacyPreservingSystemWithCache, and sends back JSON results.
- Provides error handling with traceback logs.
