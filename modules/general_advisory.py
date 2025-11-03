import os
import numpy as np
import faiss
import pickle, time
from sentence_transformers import SentenceTransformer
import ollama 

# NOTE: Since this is a module, it cannot safely import 'tracer' from 'main'.
# We must assume the 'tracer' is passed as an argument or configured globally
# in a non-circular way (e.g., in a separate config/init file).
# For this example, I will assume it is passed to the top-level function.

from opentelemetry.trace import Tracer
from opentelemetry.trace import Status, StatusCode

# Define constants locally or import them
EMBEDDING_PATH = "C:\\Users\\ashut\\OneDrive\\Desktop\\Major Project RAG\\faiss_indexes"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "llama3"

embedder = SentenceTransformer(EMBEDDING_MODEL)

# Function now accepts the tracer object
def load_state_index(state: str, tracer: Tracer):
    """
    Loads FAISS index and metadata for the given state.
    Each state should have:
      - {state}.index
      - {state}_meta.pkl
    """
    with tracer.start_as_current_span("FAISS_Load_Index") as span:
        state = state.lower()
        idx_path = os.path.join(EMBEDDING_PATH, f"{state}.index")
        meta_path = os.path.join(EMBEDDING_PATH, f"{state}_meta.pkl")

        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            span.record_exception(FileNotFoundError(f"No FAISS index for {state}"))
            span.set_status(Status(StatusCode.ERROR, "Index or metadata missing"))
            # Re-raise the exception to be caught by the calling function
            raise FileNotFoundError(f"No FAISS index or metadata found for {state}")

        index = faiss.read_index(idx_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        span.set_attribute("faiss.state", state)
        span.set_attribute("faiss.docs_loaded", len(meta["docs"]))
        span.set_attribute("faiss.index_path", idx_path)

        return index, meta


# ---- Manual FAISS + Ollama RAG ----
# Function now accepts the tracer object
def rag_pipeline(user_query: str, state: str, tracer: Tracer):
    """
    A clean FAISS + Ollama RAG pipeline with observability.
    1. Embeds query
    2. Searches FAISS index
    3. Builds context
    4. Prompts local LLM for state-aware advisory
    """
    with tracer.start_as_current_span("RAG_Pipeline", attributes={"user.state": state, "query.text": user_query}) as span:
        try:
            # Pass tracer to sub-function
            index, meta = load_state_index(state, tracer)
            docs, metadatas = meta["docs"], meta["metadatas"]

            # ---- Step 1: Embedding ----
            with tracer.start_as_current_span("Embed_Query") as subspan:
                start = time.perf_counter()
                query_vec = embedder.encode([user_query])
                subspan.set_attribute("embedding.model", EMBEDDING_MODEL)
                subspan.set_attribute("embedding.latency_s", time.perf_counter() - start)

            # ---- Step 2: FAISS Retrieval ----
            with tracer.start_as_current_span("FAISS_Search") as subspan:
                start = time.perf_counter()
                D, I = index.search(np.array(query_vec, dtype=np.float32), k=3)
                subspan.set_attribute("retrieval.latency_s", time.perf_counter() - start)
                subspan.set_attribute("retrieval.top_k", len(I[0]))
                subspan.set_attribute("retrieval.max_score", float(np.max(D))) # Log the best similarity score

            # ---- Step 3: Context Construction ----
            with tracer.start_as_current_span("Context_Build") as subspan:
                context = ""
                sources = []
                for idx in I[0]:
                    src = metadatas[idx].get("source", "unknown")
                    sources.append(src)
                    context += f"[Source: {src}]\n{docs[idx]}\n\n"
                
                # Log context and sources
                subspan.set_attribute("context.length", len(context))
                subspan.set_attribute("context.sources", sources)

            # ---- Step 4: LLM Generation ----
            with tracer.start_as_current_span("LLM_Generation") as subspan:
                start = time.perf_counter()

                prompt = f"""
                You are an experienced agricultural advisor from {state.title()}.

                Use ONLY the verified context below to answer the user's question.
                Make your answer specific to {state.title()}, clear, and actionable.
                Cite relevant source info from the context where possible.

                Context:
                {context}

                Question:
                {user_query}

                Answer:
                """

                response = ollama.chat( # Corrected: used ollama.chat for consistency
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are an expert agricultural assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response["message"]["content"].strip()
                latency = time.perf_counter() - start
                
                # Log LLM metrics
                subspan.set_attribute("llm.model", MODEL_NAME)
                subspan.set_attribute("llm.latency_s", latency)
                subspan.set_attribute("llm.output_length", len(answer))
                subspan.set_attribute("llm.prompt", prompt) # Log the full RAG prompt
                
                # The model's response is the final output
                span.set_attribute("rag.final_answer", answer[:500] + "...") # Log a snippet on the main span
                span.set_attribute("rag.status", "success")
                span.set_status(Status(StatusCode.OK)) # Explicitly set OK status on success
                return answer

        except Exception as e:
            # Catch exceptions from load_state_index or later steps
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            return f"⚠️ Error during RAG pipeline for {state}: {str(e)}"