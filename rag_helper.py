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
from openai import OpenAI
import re
import sys
import subprocess
import tempfile
import shlex

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
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents = docs,
        embedding = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval"),
        collection = atlas_collection,
        index_name = vector_search_index
    )


'''
Parameters:
    document: str - The path of the document to generate the document from : output.txt
'''
def generate_design_document(document: str):
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ['MONGODB_URI'],
        db_name + "." + collection_name,
        TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval"),
        index_name=vector_search_index,
    )
    retriever = vector_search.as_retriever(
        search_type = "similarity",
        search_kwargs = {
            "k": 10,
            "score_threshold": 0.75,
            "pre_filter": { "source": { "$eq": document } }
        }
    )
    # Define a prompt template
    template = """
    Use the following pieces of to help complete the task. Use your best judgement to complete the task.
    Use the given context to complete the task.
    {context}
    Prompt: {question}
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
    prompt = """You are a Solutions Architect specialized in designing technical design documents.  Generate a design document using the format below.

    Technology Architecture
    [Describe the anticipated infrastructure that will be required to support the application and information architecture. At this point in the project, focus on describing the technology architecture at a high level only.]

    Platform
    [Identify the target platform expected for the system (e.g., mainframe, mid-tier, or other).]

    System Hosting
    [Identify the target hosting for the system (e.g., EDC, BDC, Other).]

    Connectivity Requirements
    [Identify network connectivity requirements for the system (e.g., Internet, Extranet).]

    Modes of Operation
    [Describe the modes that the system will operate in:
    • What environments will the system need (e.g., development, test, and production)?
    • Will there be just one production instance of the system?
    • Will the old and new system run in parallel?
    • Will there be a pilot? If so, what is the success criteria and exit strategy?
    • Will the existing system be retired?
    • Should the new system convert the data from the current system?]

    Security and Privacy Architecture
    [Describe the anticipated security and privacy architecture. The purpose of this discussion is to identify the general approach to security to ensure that proper controls will be implemented into the system. This is not intended to be a detailed security design.]

    Authentication
    [Describe the basic user authentication approach to verify user identity before allowing access to the system. For example, will the system use the single-sign-on solution?]

    Authorization
    [Describe the anticipated approach for authorizing users to perform functional activity once logged into the system.]
    """
    # Prompt the chain
    answer = rag_chain.invoke(prompt)
    # Return source documents
    documents = retriever.get_relevant_documents(prompt)
    return answer

def generate_architecture_diagram_code(design_document):
    openaiClient = OpenAI()
    messages = [
        {"role": "system", "content": "You are an expert Python developer experienced with the Diagrams package. You should only give the code delimited by ```, do not give any descriptions or how to run it. The filename for the outputted image should be diagram.png, so make sure to add the parameter (, filename='diagram.png'). Give me Python code that creates an architecture diagram for the design document below:"}
        ,{
        "role": "user", "content": design_document}]
    response = openaiClient.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0)
    code_string = response.choices[0].message.content.strip()
    print("Code String: ", code_string)
    code_pattern = re.compile("```(?:python)?(.*?)```", re.DOTALL)
    code_match = code_pattern.search(code_string)
    if code_match:
        code = code_match.group(1).strip()
    else:
        print("Error: Could not find the code.")
        sys.exit(1)

    # Specify your filename here
    filename = "output.py"

    with open(filename, 'w') as file:
        file.write(code)

    manim_command = f"""python {filename}"""
    file_name_with_extension = os.path.basename(filename)
    folder_name, _ = os.path.splitext(file_name_with_extension)
    args = shlex.split(manim_command)
    try:
        # Run the Manim command
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        # Handle errors in the external commands if they fail
        print(f"An error occurred: {e}")

design_document = generate_design_document("output.txt")

print(f'Design Document: {design_document}')
print(f'Architechture Code:{generate_architecture_diagram_code(design_document)}')




