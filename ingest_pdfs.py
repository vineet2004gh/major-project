import os
import faiss
import numpy as np
import pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = "data/docs"
INDEX_DIR = "faiss_indexes"
EMBED_MODEL = "all-MiniLM-L6-v2"

os.makedirs(INDEX_DIR, exist_ok=True)
embedder = SentenceTransformer(EMBED_MODEL)
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

def extract_text_from_pdf(path):
    text = ""
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return text

def process_state(state_name):
    folder = os.path.join(BASE_DIR, state_name)
    if not os.path.isdir(folder):
        print(f"❌ Folder not found for {state_name}")
        return
    
    docs, metadatas = [], []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            print(f"📄 Reading {path}...")
            text = extract_text_from_pdf(path)
            chunks = splitter.split_text(text)
            docs.extend(chunks)
            metadatas.extend([{"source": file, "state": state_name}] * len(chunks))
    
    if not docs:
        print(f"No documents found for {state_name}")
        return

    print(f"🧠 Embedding {len(docs)} chunks for {state_name}...")
    embeddings = embedder.encode(docs, show_progress_bar=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))
    
    # Save both FAISS and metadata
    faiss.write_index(index, os.path.join(INDEX_DIR, f"{state_name}.index"))
    with open(os.path.join(INDEX_DIR, f"{state_name}_meta.pkl"), "wb") as f:
        pickle.dump({"docs": docs, "metadatas": metadatas}, f)
    
    print(f"✅ Created FAISS index for {state_name} with {len(docs)} docs.")

def main():
    for state in os.listdir(BASE_DIR):
        if os.path.isdir(os.path.join(BASE_DIR, state)):
            process_state(state)

if __name__ == "__main__":
    main()
