import streamlit as st
from dotenv import load_dotenv
import boto3
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import bedrock as bedrockEmbeddings
from langchain.vectorstores import faiss
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import bedrock as bedrock_llm
from htmlTemplates import css, bot_template, user_template

bedrock_client = boto3.client(
    service_name='bedrock-runtime', region_name='us-east-1')


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):

    bedrock_embeddings = bedrockEmbeddings.BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1", client=bedrock_client)

    vectorstore = faiss.FAISS.from_texts(
        texts=text_chunks, embedding=bedrock_embeddings)

    return vectorstore


def get_conversation_chain(vector_store):
    llm = bedrock_llm.Bedrock(
        model_id="amazon.titan-text-express-v1", client=bedrock_client)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, verbose=True, memory=memory, retriever=vector_store.as_retriever())

    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.write(response)


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat With Multiple PDFs",
                       page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with multiple PDFs :books:")

    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace(
        "{{MSG}}", "Hello Human"), unsafe_allow_html=True)
    st.write(bot_template.replace(
        "{{MSG}}", "Hello Robot"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on \'Proccess\'", accept_multiple_files=True)
        if st.button("Proccess"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
