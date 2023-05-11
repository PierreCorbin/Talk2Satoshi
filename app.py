import streamlit as st
import pinecone
import openai
import os

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone

from langchain.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
import os

     
def create_app_header():
    st.set_page_config(page_title="AiToshi", page_icon=":guardsman:")
    st.title("AiToshi")
    st.write("Welcome, pleb! Get answers to your bitcoin questions from the creator himself.")
    st.divider()
    st.write("The world of Bitcoin and cryptocurrency can be confusing and overwhelming, with new information and updates coming out every day.")
    st.write("At AiToshi, we offer a unique way to get your questions answered - by chatting with an AI version of the creator of Bitcoin himself, Satoshi Nakamoto.")
    st.write("Our AI is based on a big library of bitcoin books and all of the content written by Satoshi, making it the closest experience to chatting with Satoshi Nakamoto.")
    st.divider()
    
def display_widgets() -> str:
    query = st.text_input("Ask Satoshi a question:")
    if not query:
        st.error("Ask a question for this to work.")
    
    st.write('## Resources')
    st.write("""Here is the list of all the data used by this model: \n
    - The Price of Tomorrow, Jeff Booth, 2020 \n
    """)

    return query

def extract_query() -> str:
    query = display_widgets()

    if query:
        return query


def get_pinecone_docs(query):
    pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment=st.secrets["PINECONE_API_ENV"])

    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    chain = load_qa_chain(llm, chain_type="stuff")
    docsearch = Pinecone.from_existing_index("aitoshi", embeddings)
    docs = docsearch.similarity_search(query, include_metadata=True)

    # 4 - Le Prompt
    template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.

    Act as Satoshi Nakamoto, the creator of Bitcoin. Always start your answer by "Arigato !"

    QUESTION: {question}
    =========
    {summaries}
    =========
    FINAL ANSWER :"""

    PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

    # 5 - La chaine et le résultat
    chain = load_qa_with_sources_chain(llm=llm, chain_type="stuff", prompt=PROMPT, verbose=True)
    result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)

    return result


def main():
    create_app_header()
    
    if query := extract_query():
        with st.spinner(text="Let me think for a while..."):
            answer = get_pinecone_docs(query)

            st.success("Here is your explanation. Ask me another question!")

            st.markdown(f"**Answer:** {answer}")


if __name__ == '__main__':
    main() 
