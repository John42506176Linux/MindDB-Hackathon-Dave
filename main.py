import getpass, os, pymongo, pprint
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import together
import pymongo
from typing import List
from langchain_together.embeddings import TogetherEmbeddings
from langchain_community.document_loaders import TextLoader

load_dotenv()
# Connect to your Atlas cluster
together.api_key = os.environ['TOGETHER_API_KEY']
client = MongoClient(os.environ['MONGODB_URI'])
# Define collection and index name
db_name = "langchain_db"
collection_name = "test2"
atlas_collection = client[db_name][collection_name]
vector_search_index = "vector_index"
def embed_pdf(source: str = "output.txt"):
    loader =  TextLoader(source)
    data = loader.load()
    # Split PDF into documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    docs = text_splitter.split_documents(data)
    print("Split PDF into", len(docs), "documents")
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents = docs,
        embedding = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval"),
        collection = atlas_collection,
        index_name = vector_search_index
    )
    print("Embedded PDF into MongoDB Atlas")
    return vector_store

def query_pdf(query: str):
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ['MONGODB_URI'],
        db_name + "." + collection_name,
        TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval"),
        index_name=vector_search_index,
    )
    retriever = vector_search.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 10, "score_threshold": 0.70}
    )
    # Define a prompt template
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    """
    custom_rag_prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    # Construct a chain to answer questions on your data
    rag_chain = (
    { "context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
    )
    # Prompt the chain
    answer = rag_chain.invoke(query)
    print("Question: " + query)
    print("Answer: " + answer)
    # Return source documents
    documents = retriever.get_relevant_documents(query)
    print("\nSource documents:")
    pprint.pprint(documents)


vector_store = embed_pdf()
print("PDF embedded successfully!")
query = "What was discussed in this meeting?"
query_pdf(query)




