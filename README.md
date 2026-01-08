# ⚽ AFCON RAG: AI-Powered Football Analyst

An end-to-end **Retrieval-Augmented Generation (RAG)** application designed to answer complex questions about the Africa Cup of Nations (AFCON) from 2006 to 2026. The system uses a specialized Knowledge Base built from PDF documents to provide factual, grounded responses.

## 🚀 Features

* **Semantic Search:** Uses `BGE-base-en-v1.5` embeddings for high-accuracy document retrieval instead of simple keyword matching.
* **Knowledge Base:** Ingests private PDF data about AFCON history, records, and winners.
* **Hybrid Architecture:** * **Backend:** FastAPI for high-performance API serving.
  * **Frontend:** Streamlit for a clean, interactive user interface.
* **LLM Integration:** Powered by **ChatGroq** (openai/gpt-oss-120b) for lightning-fast and intelligent response generation.

---

## 🏗️ Technical Stack

* **Orchestration:** LangChain (LCEL)
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace `BAAI/bge-base-en-v1.5`
* **LLM:** Groq API
* **Web Frameworks:** FastAPI & Streamlit

---

## 🛠️ Installation & Setup

### 1. Clone the repository

**Bash**

```
git clone https://github.com/yourusername/afcon-rag.git
cd afcon-rag
```

### 2. Install dependencies

**Bash**

```
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the root directory and add your Groq API key:

**Plaintext**

```
GROQ_API_KEY=your_api_key_here
```

### 4. Knowledge Base Setup

Ensure your `AFCON-English.pdf` is in the directory. Run the initial ingestion (or ensure the `afcon_chroma_db` folder exists).

---

## 🚦 How to Run

You need to run the Backend and Frontend in separate terminals:

**Terminal 1: Backend (FastAPI)**

**Bash**

```
uvicorn main:app --reload
```

**Terminal 2: Frontend (Streamlit)**

**Bash**

```
streamlit run app.py
```

---

## 📝 Usage Examples

The system is specialized in AFCON data between 2006 and 2026:

* *Question:* "Who won AFCON 2012 and what was their story?"
* *Question:* "Which country won the title 3 times in a row?"
* *Question:* "Who is the all-time top scorer mentioned in the docs?"

---

## 🔒 Security Note

* The `.env` file and `afcon_chroma_db/` are ignored via `.gitignore` to protect sensitive API keys and local data structures.
* All LLM responses are strictly grounded in the provided context to prevent hallucinations.

---

Would you like me to help you set up a **GitHub Action** to automate the testing of this code?
