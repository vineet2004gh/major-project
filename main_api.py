import os
import sys
import io
import time
import pickle
import numpy as np
import faiss
import subprocess
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from location_api import get_user_state
from modules import general_advisory, disease_classification, yield_prediction
from google import genai
from google.genai import types
# ---- UTF-8 console fix for Windows ----
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ---- Load environment variables ----
load_dotenv()

# ---- Initialize OpenTelemetry ----
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
exporter = OTLPSpanExporter()
processor = BatchSpanProcessor(exporter)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("AgriAdvisor-Tracer")

# ---- Initialize Gemini ----
try:
    client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
    print("✅ Gemini Client Initialized.")
except Exception as e:
    print(f"🚨 Error initializing Gemini client: {e}")
    client = None

# ---- App and Embedding Model ----
app = FastAPI(title="AgriAdvisor - State-Aware RAG")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_DIR = "faiss_indexes"

# ---- Router ----
def route_query(user_query: str):
    system_prompt = """
    You are an EXTREMELY STRICT and specialized Query Router for agricultural questions.
    Your SOLE purpose is to classify the user's query into one of three categories:
    1. yield   → crop output, productivity, harvest estimation.
    2. disease → crop health, pests, infections, nutrient issues.
    3. general → irrigation, fertilizer (non-deficiency), government schemes, weather, machinery, etc.
    STRICT RULES:
    • Respond with exactly one token: yield, disease, or general.
    • If ambiguous: GENERAL > DISEASE > YIELD.
    """

    if client is None:
        print("🚨 Gemini client unavailable. Defaulting to 'general'.")
        return "general"

    with tracer.start_as_current_span("RouterLLM") as span:
        span.set_attribute("llm.model", "gemini-2.5-flash")
        span.set_attribute("llm.system_prompt", system_prompt)
        span.set_attribute("llm.user_query", user_query)

        start = time.perf_counter()
        route_raw = "general"

        try:
            # Use the Gemini model to classify the query
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt),
                contents=user_query
            )
            route_raw = response.text
        except Exception as e:
            print(f"🚨 Gemini API Error: {e}")

        elapsed = time.perf_counter() - start
        span.set_attribute("router.latency_s", elapsed)

    route = (route_raw or "").strip().lower()
    with tracer.start_as_current_span("RouterPostProcess") as span:
        span.set_attribute("query.route_result", route)

    print(f"🧭 Route chosen: {route} ({elapsed:.3f}s)")

    if "disease" in route:
        return "disease"
    if "yield" in route:
        return "yield"
    if "general" in route:
        return "general"
    return "general"

# ---- State Detector ----
def detect_state():
    with tracer.start_as_current_span("LocationAPI") as span:
        start = time.perf_counter()
        state = get_user_state()
        elapsed = time.perf_counter() - start
        span.set_attribute("location_api.latency_s", elapsed)
        span.set_attribute("user.state", state)
        return state

# ---- Main Orchestration ----
def main():
    query = input("Enter your question: ")

    with tracer.start_as_current_span("UserQueryProcessing") as parent_span:
        parent_span.set_attribute("user.full_query", query)

        state = detect_state()
        route = route_query(query)

        parent_span.set_attribute("user.state", state)
        parent_span.set_attribute("query.route", route)

        print(f"🧭 Route: {route}")
        print(f"🌾 State: {state}")

        with tracer.start_as_current_span("ModuleExecution") as span:
            start = time.perf_counter()
            span.set_attribute("executed.module", route)

            try:
                if route == "general":
                    answer = general_advisory.rag_pipeline(query, state, tracer)
                elif route == "yield":
                    answer = yield_prediction.handle_yield_prediction(query)
                elif route == "disease":
                    answer = disease_classification.handle_disease_classification(query)
                else:
                    answer = "I couldn’t categorize your request."
                    span.set_attribute("execution.error", "Uncategorized route")
            except Exception as e:
                answer = f"⚠️ Error during module execution: {e}"

            elapsed = time.perf_counter() - start
            span.set_attribute("execution.latency_s", elapsed)
            parent_span.set_attribute("final.answer", str(answer)[:200] + "...")

    print("\n🧠 AI Advisor Output:\n", str(answer))
    print("\n📊 Trace captured in Phoenix dashboard.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication interrupted.")
    finally:
        provider.shutdown()
        print("✅ OpenTelemetry Tracer Shutdown. Final spans exported.")
        time.sleep(1)
