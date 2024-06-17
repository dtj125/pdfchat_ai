from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large",)

# Create a FAISS instance for vector database from 'data'
vectordb = FAISS.from_documents(documents=data,
                                 embedding=instructor_embeddings)

#Save the Vector Store DB
vectordb.save_local("macchu_picchu_vd")



#//// GPT version

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_vector_store(text_chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Builds a vector store using FAISS for the given text chunks.
    
    Args:
    text_chunks (list of str): List of text chunks.
    model_name (str): Name of the Hugging Face model to use for embeddings.
    
    Returns:
    faiss.IndexFlatL2: FAISS index for the vector store.
    """
    # Define embedding model
    embedding_model = SentenceTransformer(model_name)
    
    # Embed text chunks
    embedded_vectors = embedding_model.encode(text_chunks)
    
    # Build FAISS index
    d = embedded_vectors.shape[1]  # Dimension of the vectors
    index = faiss.IndexFlatL2(d)  # Instantiate FAISS index
    index.add(embedded_vectors)  # Add vectors to the index
    
    return index

# Example usage:
if __name__ == "__main__":
    # Example text chunks
    text_chunks = ["First chunk of text.", "Second chunk of text.", "Third chunk of text."]
    
    # Build vector store
    vector_store = build_vector_store(text_chunks)
    
    # You can now use 'vector_store' to perform similarity searches or other operations
