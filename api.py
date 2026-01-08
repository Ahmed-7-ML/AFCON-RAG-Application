from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import Libraries (that are helpful from notebook)
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ------------------------ Dependencies ----------------------------------
from config import BGEEmbeddings, llm, prompt_template

app = FastAPI(title="AFCON RAG API")

vector_store = Chroma(persist_directory='./afcon_chroma_db',
            embedding_function=BGEEmbeddings())
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# ------------------------ Application Programming Interface ----------------------------------
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        response = rag_chain.invoke(request.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
