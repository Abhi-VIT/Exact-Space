# This cell contains the complete code for the minimal runnable RAG prototype.
# You can copy this code and save it as a Python notebook (e.g., rag_demo.ipynb)
# in the 'prototype/' folder.

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# --- Configuration ---
docs_dir = "docs"
chunk_size = 500
chunk_overlap = 100
embedding_model_name = 'all-MiniLM-L6-v2'
retrieval_k = 5

# --- 1. Document Ingestion and Chunking ---

print("--- Document Ingestion and Chunking ---")

if not os.path.exists(docs_dir):
    os.makedirs(docs_dir)
    print(f"Created directory: {docs_dir}. Please place your sample PDF documents here.")
    documents = [] # No documents to process if directory was just created
else:
    print(f"Using document directory: {docs_dir}")
    documents = []
    for filename in os.listdir(docs_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(docs_dir, filename)
            try:
                with fitz.open(filepath) as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    documents.append({"filename": filename, "content": text})
                print(f"Processed document: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if not documents:
    print("No PDF documents found in the 'docs' directory. Please add some sample PDFs to proceed with chunking and embedding.")
    chunks = [] # No chunks if no documents
else:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.create_documents([doc["content"]])
        for i, chunk in enumerate(doc_chunks):
            chunks.append({
                "filename": doc["filename"],
                "chunk_id": f"{doc['filename']}_chunk_{i+1}",
                "content": chunk.page_content,
                "source": f"{doc['filename']}_chunk_{i+1}"
            })
    print(f"Created {len(chunks)} chunks from {len(documents)} documents.")
    if chunks:
        print("First chunk example:")
        print(chunks[0]['content'])
        print("Source:", chunks[0]['source'])

# --- 2. Embedding and Indexing (Vector Database) ---

print("\n--- Embedding and Indexing ---")

if chunks:
    embedding_model = SentenceTransformer(embedding_model_name)
    print(f"Embedding model '{embedding_model_name}' loaded.")

    chunk_contents = [chunk["content"] for chunk in chunks]
    chunk_embeddings = embedding_model.encode(chunk_contents)
    print(f"Created embeddings for {len(chunk_embeddings)} chunks.")

    chunk_embeddings = np.array(chunk_embeddings).astype('float32')

    faiss_index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    print(f"FAISS index created with dimension: {chunk_embeddings.shape[1]}")

    faiss_index.add(chunk_embeddings)
    print(f"Added {faiss_index.ntotal} embeddings to the index.")

    indexed_chunks = chunks # Store chunks with index
    print("Embedding and indexing complete.")
else:
    print("Skipping embedding and indexing as no chunks were created.")
    faiss_index = None
    indexed_chunks = []

# --- 3. Retrieval ---

print("\n--- Retrieval ---")

def retrieve_chunks(query, embedding_model, faiss_index, indexed_chunks, k=5):
    if faiss_index is None or not indexed_chunks:
        print("Index is not available or no chunks are indexed.")
        return []

    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = faiss_index.search(query_embedding, k)
    retrieved_chunks = [indexed_chunks[i] for i in indices[0]]
    return retrieved_chunks

# --- 4. Answer Generation with LLM (Placeholder) ---

print("\n--- Answer Generation (Placeholder) ---")

def generate_answer_with_citations(query, retrieved_chunks, llm_model):
    if not retrieved_chunks:
        return "No relevant information found in the documents."

    context = "\n\n".join([f"Source: {chunk['source']}\n{chunk['content']}" for chunk in retrieved_chunks])
    prompt = f"Based on the following technical documentation, answer the query:\n\n{context}\n\nQuery: {query}\n\nAnswer:"

    # --- Placeholder for LLM interaction ---
    # Replace this with code to call your LLM.
    # Example using a hypothetical local LLM:
    # from transformers import pipeline
    # llm = pipeline("text-generation", model="your-local-llm-name")
    # generated_text = llm(prompt, max_length=500)[0]['generated_text']
    # --- End of Placeholder ---

    # For the prototype, we'll just show the context that would be used
    generated_text = f"Placeholder Answer: (Replace this with LLM generated text based on the context)\n\nRetrieved context used:\n{context}"

    return generated_text

# --- Example Usage ---

if faiss_index is not None and indexed_chunks:
    query = "What does a sudden draft drop indicate?" # replace it with input() for dynamic queries
    print(f"Query: '{query}'")
    retrieved_chunks = retrieve_chunks(query, embedding_model, faiss_index, indexed_chunks, k=retrieval_k)

    print(f"\nTop {len(retrieved_chunks)} retrieved chunks:")
    for i, chunk in enumerate(retrieved_chunks):
         print(f"--- Chunk {i+1} (Source: {chunk['source']}) ---")
         print(chunk['content'])

    # Example LLM interaction (using placeholder)
    llm_model = None # Replace with your loaded LLM
    answer = generate_answer_with_citations(query, retrieved_chunks, llm_model)
    print("\n--- Generated Answer ---")
    print(answer)
else:
    print("\nSkipping retrieval and answer generation as no documents were processed.")

# --- 5. Evaluation Outline ---
print("\n--- Evaluation Outline ---")
print("Refer to the 'Evaluation (Conceptual Outline)' section in the notebook/notes.md for details on how to evaluate the prototype.")