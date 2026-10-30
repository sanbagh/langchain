import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_ollama import OllamaEmbeddings

load_dotenv()

if __name__ == "__main__":
    
    loader = UnstructuredLoader(file_path="mediumblog1.txt",chunking_strategy="basic", max_characters=1000000)
    document = loader.load()
    
    text_splitters = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitters.split_documents(document)
    print(f"created {len(texts)} chunks")
    
    embed = OllamaEmbeddings(model="nomic-embed-text")
    
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["INDEX_NAME"]
    
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
    )
        print(f"Created index: {index_name}")
    else:
        print(f"Index already exists: {index_name}")
        
    print("Ingesting....")
    PineconeVectorStore.from_documents(texts,embed,index_name=os.environ["INDEX_NAME"])
    print("finish")