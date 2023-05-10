import streamlit as st
import pinecone
import openai
import os

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone

     
def create_app_header():
    st.set_page_config(page_title="Talk2Satoshi", page_icon=":guardsman:")
    st.title("Talk2Satoshi")
    st.write("Welcome, pleb! Get answers to your bitcoin questions from the creator himself.")
    st.divider()
    st.write("The world of Bitcoin and cryptocurrency can be confusing and overwhelming, with new information and updates coming out every day.")
    st.write("At Talk2Satoshi, we offer a unique way to get your questions answered - by chatting with an AI version of the creator of Bitcoin himself, Satoshi Nakamoto.")
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
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
    chain = load_qa_chain(llm, chain_type="stuff")
    docsearch = Pinecone.from_existing_index("talk2satoshi", embeddings)
    docs = docsearch.similarity_search(query, include_metadata=True)

    print(f"Answer as if you were Satoshi Nakamoto:{query}")
    return chain.run(input_documents=docs, question=f"Answer pretending to be Satoshi Nakamoto:{query}")


def main():
    create_app_header()
    
    if query := extract_query():
        with st.spinner(text="Let me think for a while..."):
            answer = get_pinecone_docs(query)

            st.success("Here is your explanation. Ask me another question!")

            st.markdown(f"**Answer:** {answer}")


if __name__ == '__main__':
    main() 
