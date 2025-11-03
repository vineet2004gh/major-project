import subprocess
import time
import csv
import os
import sys
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 10 representative queries
queries = [
    "What crop gives the highest yield during monsoon in Maharashtra?",
    "My wheat leaves have yellow spots, what disease is it?",
    "How much yield can I expect if I plant rice this month?",
    "What fertilizers should I use for better growth of sugarcane?",
    "Are there any government subsidies for drip irrigation in Punjab?",
    "How can I prevent fungal infection in my paddy fields?",
    "What is the expected rainfall for Kharif season this year?",
    "What measures can improve maize yield in dry regions?",
    "Is there a pest resistant variety of cotton suitable for Uttar Pradesh?",
    "Which crops are best suited for sandy soil in Maharashtra?"
]

REPORT_PATH = "benchmark_report.csv"

def run_benchmark():
    results = []
    for i, q in enumerate(queries, 1):
        print(f"\n[{i}/10] Running query: {q}")
        start = time.time()
        process = subprocess.Popen(
            [sys.executable, "main_api.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True   
        )

        try:
            out, err = process.communicate(input=q + "\n", timeout=900)
            elapsed = time.time() - start
        except subprocess.TimeoutExpired:
            process.kill()
            out, err = "", "Timeout"
            elapsed = -1

        print(f"⏱ Time: {elapsed:.2f}s")

        results.append({
            "query": q,
            "time_s": elapsed,
            "stdout": out.strip(),
            "stderr": err.strip()
        })

    # Write CSV report
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "time_s", "stdout", "stderr"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Benchmark report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    run_benchmark()
