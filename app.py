from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from config import BGEEmbeddings, llm, prompt_template
import base64
import streamlit as st
# import requests

st.set_page_config(page_title="AFCON AI Assistant", page_icon="⚽")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(bin_file):
    bin_str = get_base64_of_bin_file(bin_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_png_as_page_bg('bg.jpg')

# Initialize vectorstore
vector_store = Chroma(persist_directory='./afcon_chroma_db', embedding_function=BGEEmbeddings())
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

st.title("⚽ AFCON Chatbot")
st.markdown("Ask me anything about the Africa Cup of Nations (2006-2026)")

user_query = st.text_input("Type your question here:", placeholder="Who won AFCON in 2006?")

if st.button("Submit"):
    if user_query:
        with st.spinner("We are currently searching the documents and generating the answer..."):
            try:
                # payload = {"question": user_query}
                # response = requests.post(
                #     "http://127.0.0.1:8000/ask", json=payload)

                # if response.status_code == 200:
                #     answer = response.json().get("answer")
                    answer = rag_chain.invoke(user_query)
                    st.success(answer)
                # else:
                #     st.error("An error occurred while connecting to the server.")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question first.")