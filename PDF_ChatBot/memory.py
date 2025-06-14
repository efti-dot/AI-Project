import os
from fastapi import FastAPI, HTTPException
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.chains import RetrievalQA

Path = "Book/"
def load_pdf(Book):
    loader = DirectoryLoader(Book, glob='*.pdf', loader_cls=PyPDFLoader)

    documents = loader.load()
    return documents

documents = load_pdf(Book = Path)
print("len :", len(documents))


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,)
docs = text_splitter.split_documents(documents)
print("Chunks created:", len(docs))

#OpenAIembedding
#embeddings = OpenAIEmbeddings()
#vectorstore = FAISS.from_documents(docs, embeddings)

#Huggingface
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

#model
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_length=512,
    temperature=0.5,
    top_p=1,
    repetition_penalty=1.1
)

local_llm = HuggingFacePipeline(pipeline=qa_pipeline)

qa = RetrievalQA.from_chain_type(
    llm=local_llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

#query = "What is the main topic discussed in the introduction?"
query = "How to cure Cancer?"
result = qa(query)

print("Answer:", result['result'])

