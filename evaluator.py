import pandas as pd
from tqdm import tqdm
import time
import json
import re
from models.scout_api import call_scout_model 

RATE_LIMIT = 20       # Number of requests before cooldown
COOLDOWN_TIME = 60

def evaluation_prmp(prompt, answer_a, answer_b):
    return f"""
You're a highly capable evaluator model. Given a prompt and two AI-generated answers, compare them on:
1. Correctness
2. Depth and clarity
3. Presence of hallucinations

Return:
-Better answer (1: TinyLlama /2: Llama-4 Scout)
-Out of both, which answer is hallucinated? (1: TinyLlama /2: Llama-4 Scout)
-Reason for the decision

PROMPT:
{prompt}

Answer A:
{answer_a}

Answer B:
{answer_b}

The answer should only be the JSON object, without any additional text or explanation.
The JSON object should be formatted as mentioned below:
{{
    "better_answer": "1 or 2",
    "hallucinated_answer": "1 or 2",
    "reason": "string"
}}
If neither is hallucinated, hallucinated_answer should be "0".
If both are equally good, better_answer "1".
Dont't generate  any text except the JSON formatted answer. 
""".strip()

def extract_json_from_response(response_text):
    try:
        # Extract JSON block inside markdown (```json ... ```)
        match = re.search(r"```json\s*({[\s\S]*?})\s*```", response_text)
        if not match:
            # Fallback to plain curly braces
            match = re.search(r"({[\s\S]*?})", response_text)
        
        if match:
            json_str = match.group(1)
            # Fix invalid backslashes (e.g. \( → ( ) for safe JSON parsing)
            json_str = json_str.replace("\\(", "(").replace("\\)", ")").replace('\\"', '"')
            return json.loads(json_str)
    except Exception as e:
        print(f"⚠️ JSON extraction failed: {e}")
    return None


def run_evaluation(csv1_path, csv2_path, output_path):
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    request_count = 0

    assert len(df1) == len(df2), "CSV row lengths don't match!"

    results = []

    for idx in tqdm(range(len(df1))):
        if request_count >= RATE_LIMIT:
            print(f"⏸ Rate limit hit. Waiting {COOLDOWN_TIME} seconds...")
            time.sleep(COOLDOWN_TIME)
            request_count = 0
        
        row1 = df1.iloc[idx]
        row2 = df2.iloc[idx]

        prompt = row1["Prompt"]
        answer_a = row1["Response"]
        answer_b = row2["Response"]

        if row1["Difficulty"].lower() != "hard":
            results.append({
                "index": idx,
                "prompt": prompt,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "judgment": "0",
                "hallucination": "0",
                "reason": "-",
            })
            continue

        eval_prompt = evaluation_prmp(prompt, answer_a, answer_b)
        raw_response = call_scout_model(eval_prompt)
        response = raw_response["response"]

        print(f"Raw response at prompt {idx}:\n{response}")

        evaluation = extract_json_from_response(response)
        if evaluation is None:
            print(f"❌ Failed to parse response at index {idx}. Logging fallback entry.")
            results.append({
                "index": idx,
                "prompt": prompt,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "judgment": "parse_error",
                "hallucination": "parse_error",
                "reason": "Could not extract valid JSON from model output.",
            })
            continue

        best_answer = evaluation.get("better_answer", "parse_error")
        has_hallucination = evaluation.get("hallucinated_answer", "parse_error")
        reason = evaluation.get("reason", "No reason provided.")

        request_count += 1

        results.append({
            "index": idx,
            "prompt": prompt,
            "answer_a": answer_a,
            "answer_b": answer_b,
            "judgment": best_answer,
            "hallucination": has_hallucination,
            "reason": reason,
        })

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"✅ Evaluation complete. Saved to {output_path}")

if __name__ == "__main__":
    run_evaluation(
        "logs/arc_log_mistral_only_set1_20250415_165611.csv",
        "logs/arc_log_scout_only_set1_20250415_170744.csv",
        "logs/scout_comparison_results_SET1.csv"
    )
