#!/usr/bin/env python3
"""
Script to populate Pinecone index with MedReferral knowledge base
Run this once to set up your Pinecone index with the knowledge base data
"""

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient

# Load environment variables
load_dotenv()

def populate_pinecone_index():
    """Populate Pinecone index with knowledge base data"""
    
    # Get API keys
    google_api_key = os.getenv("GEMINI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    if not google_api_key:
        print("Error: GEMINI_API_KEY not found in environment variables")
        return False
        
    if not pinecone_api_key:
        print("Error: PINECONE_API_KEY not found in environment variables")
        return False
    
    # Check if knowledge base file exists
    txt_path = "website_knowledge_base.txt"
    if not os.path.exists(txt_path):
        print(f"Error: Knowledge base file not found: {txt_path}")
        return False
    
    try:
        print("[*] Starting Pinecone index population...")
        
        # Initialize Pinecone client
        pc = PineconeClient(api_key=pinecone_api_key)
        index_name = "ellie"
        
        # Check if index exists
        if index_name not in pc.list_indexes().names():
            print(f"Error: Pinecone index '{index_name}' not found")
            print("Please create the index first or check the index name")
            return False
        
        print(f"[*] Found existing Pinecone index: {index_name}")
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=google_api_key
        )
        
        # Load and process documents
        print("[*] Loading knowledge base documents...")
        loader = TextLoader(txt_path, encoding="utf-8")
        docs = loader.load()
        
        print("[*] Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        docs_chunks = splitter.split_documents(docs)
        
        print(f"[*] Created {len(docs_chunks)} document chunks")
        
        # Connect to Pinecone index
        vectordb = Pinecone.from_existing_index(index_name, embeddings)
        
        # Clear existing data (optional - remove this if you want to keep existing data)
        print("[*] Clearing existing data from index...")
        # vectordb.delete(delete_all=True)
        
        # Add documents to Pinecone
        print("[*] Adding documents to Pinecone index...")
        vectordb.add_documents(docs_chunks)
        
        print("[*] Successfully populated Pinecone index with knowledge base!")
        print(f"[*] Added {len(docs_chunks)} document chunks to index '{index_name}'")
        
        return True
        
    except Exception as e:
        print(f"Error populating Pinecone index: {e}")
        return False

if __name__ == "__main__":
    print("MedReferral Pinecone Index Population Script")
    print("=" * 50)
    
    success = populate_pinecone_index()
    
    if success:
        print("\n✅ Pinecone index populated successfully!")
        print("You can now use the chatbot with full AI capabilities.")
    else:
        print("\n❌ Failed to populate Pinecone index.")
        print("Check your API keys and try again.")
