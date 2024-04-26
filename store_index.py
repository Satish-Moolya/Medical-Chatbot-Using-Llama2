from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY') 
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV') 

#print(PINECONE_API_KEY)
#print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")

#each chunk 500 tokens, total chunks = 7020
text_chunks = text_split(extracted_data)

embeddings = download_hugging_face_embeddings()

index_name = "medical-bot"
docsearch = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)

