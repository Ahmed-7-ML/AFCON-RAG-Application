import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

load_dotenv()

class BGEEmbeddings(Embeddings):
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _embed(self, texts):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().tolist()

    def embed_documents(self, texts):
        texts = ["Represent this passage for retrieval: " + t for t in texts]
        return self._embed(texts)

    def embed_query(self, text):
        text = "Represent this question for searching: " + text
        return self._embed([text])[0]

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """
            You are an expert football analyst specialized in the Africa Cup of Nations (AFCON).
            Your task is to answer questions ONLY using the provided retrieved context from the AFCON documents.
            Do NOT use any external knowledge or assumptions.
            If the answer is not explicitly mentioned in the context, respond with:
            "I could not find this information in the provided AFCON documents."

            When answering:
            - Be concise, factual, and accurate
            - Use clear football terminology
            - Reference years, teams, and records exactly as stated
            - If possible, mention the tournament year in your answer

            Your responses must be grounded strictly in the AFCON data.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Context: {context}

            Question: {question}
            """
        )
    ]
)

api_key = os.getenv("GROQ_API_KEY")
os.environ['GROQ_API_KEY'] = api_key
llm = ChatGroq(model_name='openai/gpt-oss-120b')
