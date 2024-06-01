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

load_dotenv()
# Connect to your Atlas cluster
client = MongoClient(os.environ['MONGODB_URI'])
# Define collection and index name
db_name = "langchain_db"
collection_name = "test"
atlas_collection = client[db_name][collection_name]
vector_search_index = "vector_index"