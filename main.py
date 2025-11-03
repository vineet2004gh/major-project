import os, faiss, pickle, numpy as np
from fastapi import FastAPI
import subprocess
from sentence_transformers import SentenceTransformer
from location_api import get_user_state
# Ensure these modules exist or are mocked for a runnable example
from modules import general_advisory, disease_classification, yield_prediction 
import time
import ollama
import sys, io
# ---- UTF-8 console fix for Windows ----
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# import phoenix as px # Not strictly needed if server is external, but can be kept for future use
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.semconv.trace import SpanAttributes # Optional, for better attribute names

# ---- Initialize OpenTelemetry (NO Phoenix Server Launch) ----
# NOTE: This setup relies on the Phoenix server being launched externally (e.g., via start_phoenix.py)

provider = TracerProvider()
# Exporter defaults to http://localhost:4318/v1/traces, connecting to the external collector.
exporter = OTLPSpanExporter()
processor = BatchSpanProcessor(exporter)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("AgriAdvisor-Tracer") # The tracer object for your app

# ---- Setup ----
app = FastAPI(title="AgriAdvisor - State-Aware RAG")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_DIR = "faiss_indexes"


# ---- Dummy Modules (Keeping the original structure for completeness) ----
# Note: These lines from the original post are placeholders and should be removed 
# if the actual imported modules are meant to be used.
def yield_prediction_module(query):
    return "🌾 Yield prediction module coming soon!"

def disease_classifier_module(query):
    return "🧬 Crop disease detection module under development!"

# ---- Router ----
def route_query(user_query):
    system_prompt = """
    You are an **EXTREMELY STRICT** and specialized Query Router for agricultural questions.
    Your SOLE purpose is to classify the user's query into one of the three predefined categories.
    CATEGORIES and DEFINITIONS:
    1.  [yield]   → Related to crop output, harvest estimation, productivity, quantity, or future production forecasting.
    2.  [disease] → Related to crop health, pest presence, insects, leaf spots, fungal/viral infection, nutrient deficiencies, or symptoms of sickness.
    3.  [general] → All other topics, including irrigation, fertilizer use (non-deficiency), government schemes, weather, soil advice, machinery, market prices, or basic farming practices.
    ***STRICT OUTPUT INSTRUCTION (NON-NEGOTIABLE)***
    1.  You MUST respond with ONLY ONE TOKEN.
    2.  The only acceptable tokens are: yield, disease, or general.
    3.  DO NOT include any punctuation, brackets, quotes, explanations, reasoning, or any other text whatsoever.
    4.  If a query is ambiguous, prioritize classification in this order: DISEASE > YIELD > GENERAL.
    Example 1:
    given question: "My tomato plants have yellow spots on the lower leaves, is this a virus?"
    Output: disease
    Example 2:
    given question: "What is the projected rice output per hectare this year?"
    Output: yield
    Example 3:
    given question: "How often should I water my corn during the summer months?"
    Output: general
    """

    # Note: 'tracer' is globally available because of the module-level setup above
    with tracer.start_as_current_span("RouterLLM") as span:
        # Log LLM specific attributes (model, prompt)
        span.set_attribute("llm.model", "llama3")
        span.set_attribute("llm.system_prompt", system_prompt)
        span.set_attribute("llm.user_query", user_query)

        start = time.perf_counter()
        response = ollama.chat(
            model='llama3',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        elapsed = time.perf_counter() - start
        
        # Log latency and LLM output
        span.set_attribute("router.latency_s", elapsed)

    route = response['message']['content'].strip().lower()
    # Log the resulting route on the router span
    # Note: If you received the 'Setting attribute on ended span' error, 
    # it means the span closed before you set the attribute. We move this inside the block.
    # However, since you didn't, we keep it here for clean post-processing logic. 
    # The actual attribute setting should technically be inside the 'with' block.
    # Let's adjust for strict correctness:
    with tracer.start_as_current_span("RouterPostProcess") as span:
        span.set_attribute("query.route_result", route) 

    print(f"🧭 Route chosen: {route} ({elapsed:.3f}s)")
    # Validate and normalize the router output
    valid_routes = ("yield", "disease", "general")

    # Return the first valid route that appears anywhere in the LLM output.
    route_text = route or ""
    if "disease" in route_text:
        return "disease"
    if "yield" in route_text:
        return "yield"
    if "general" in route_text:
        return "general"
    # Fallback
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

    # The main trace/span for the user's request
    with tracer.start_as_current_span("UserQueryProcessing") as parent_span:
        parent_span.set_attribute("user.full_query", query)
        
        state = detect_state()
        route = route_query(query)
        
        # Set final attributes on the main span
        parent_span.set_attribute("user.state", state)
        parent_span.set_attribute("query.route", route)

        print(f"🧭 Route: {route}")
        print(f"🌾 State: {state}")

        # Execute the correct module
        with tracer.start_as_current_span("ModuleExecution") as span:
            start = time.perf_counter()
            span.set_attribute("executed.module", route)

            if route == "general":
                # CRUCIAL FIX: Pass the tracer object to the module to avoid circular import issues
                answer = general_advisory.rag_pipeline(query, state, tracer) 
            elif route == "yield":
                # Assuming these modules also need the tracer if they use OpenTelemetry
                answer = yield_prediction.handle_yield_prediction(query) 
            elif route == "disease":
                answer = disease_classification.handle_disease_classification(query)
            else:
                answer = "I couldn’t categorize your request."
                span.set_attribute("execution.error", "Uncategorized route")


            elapsed = time.perf_counter() - start
            span.set_attribute("execution.latency_s", elapsed)
            parent_span.set_attribute("final.answer", answer[:200] + "...")
            

    print("\n🧠 AI Advisor Output:\n", answer)
    print("\n📊 Trace captured in Phoenix dashboard.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication interrupted.")
    finally:
        # CRUCIAL FIX: Shutdown the provider to force the final batch export
        provider.shutdown()
        print("✅ OpenTelemetry Tracer Shutdown. Final spans exported.")
        # Give the exporter a moment to complete
        time.sleep(1)