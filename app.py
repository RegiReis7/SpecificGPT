import streamlit as st

def main():
    st.set_page_config(page_title="Chat With Multiple PDFs", page_icon=":books:")

    st.header("Chat with multiple PDFs :books:")

    st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your Documents")
        st.file_uploader("Upload your PDFs here and click on \'Proccess\'")
        st.button("Proccess")


if __name__ == '__main__':
    main()