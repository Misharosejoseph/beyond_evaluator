# llm_evaluation_pipeline.py

import json
import time
import os

# -----------------------------
# STEP 1: Load JSON Inputs
# -----------------------------
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# -----------------------------
# STEP 2: Evaluate Relevance & Completeness
# -----------------------------
def check_relevance_and_completeness(response, context_vectors):
    response_text = response.lower()
    score = 0

    for ctx in context_vectors:
        if ctx.lower() in response_text:
            score += 1

    relevance_ratio = score / len(context_vectors) if context_vectors else 1

    return relevance_ratio

# -----------------------------
# STEP 3: Check Hallucination / Factual Accuracy
# -----------------------------
def check_hallucination(response, context_vectors):
    hallucinated_items = []

    context_text = " ".join(context_vectors).lower()

    for word in response.split():
        if word.lower() not in context_text:
            hallucinated_items.append(word)

    return hallucinated_items[:5]

# -----------------------------
# STEP 4: Latency & Cost Tracking
# -----------------------------
def measure_latency(function, *args):
    start = time.time()
    result = function(*args)
    end = time.time()
    return result, end - start

# Simulated token cost
TOKEN_COST_PER_WORD = 0.00001

def estimate_cost(response):
    tokens = len(response.split())
    return tokens * TOKEN_COST_PER_WORD

# -----------------------------
# STEP 5: Run Complete Evaluation
# -----------------------------
def evaluate_llm(chat_json_path, context_json_path):
    chat = load_json(chat_json_path)

    # Resolve context file: try provided path, then fallback to common alternatives
    if os.path.exists(context_json_path):
        context = load_json(context_json_path)
    else:
        alt_paths = ["contexts.json", "context.json", "contexts.json"]
        found = None
        for p in alt_paths:
            if os.path.exists(p):
                found = p
                break
        if found:
            context = load_json(found)
        else:
            raise FileNotFoundError(f"Context JSON not found. Tried: {context_json_path} and {alt_paths}")

    last_user_message = chat["messages"][-1]["content"]
    ai_response = chat["assistant_response"]
    context_vectors = context.get("vectors", [])

    relevance_score = check_relevance_and_completeness(ai_response, context_vectors)
    hallucinations = check_hallucination(ai_response, context_vectors)
    cost = estimate_cost(ai_response)

    return {
        "relevance_score": relevance_score,
        "hallucinations": hallucinations,
        "estimated_cost_usd": cost,
    }

# -----------------------------
# STEP 6: Main Entry
# -----------------------------
if __name__ == "__main__":
    output = evaluate_llm("chat.json", "context.json")
    print("Evaluation Results:\n", json.dumps(output, indent=2))
