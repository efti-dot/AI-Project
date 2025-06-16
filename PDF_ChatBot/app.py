import streamlit as st
from memory import get_qa_chain

st.title("PDF Chatbot")

qa = get_qa_chain()

question = st.text_input("Ask a question about your document:")
if question:
    with st.spinner("Thinking..."):
        result = qa.invoke({"query": question})
        st.success("Answer:")
        st.write(result)
