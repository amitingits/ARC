import pandas as pd
from tqdm import tqdm
from models.scout_api import call_scout_model 

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

Answer in only json format:
{
    "better_answer": "1 or 2",
    "hallucinated_answer": "1 or 2",
    "reason": "string"
}
If neither is hallucinated, hallucinated_answer should be "0".
If both are equally good, better_answer "1".
""".strip()

def run_evaluation(csv1_path, csv2_path, output_path):
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    assert len(df1) == len(df2), "CSV row lengths don't match!"

    results = []

    for idx in tqdm(range(len(df1))):
        row1 = df1.iloc[idx]
        row2 = df2.iloc[idx]

        if row1["difficulty"].lower() != "hard" or row1["source"].lower() != "scout":
            results.append({
            "index": idx,
            "prompt": prompt,
            "answer_a": answer_a,
            "answer_b": answer_b,
            "judgment": "0",
            "hallucination": "0 ",
        })
            continue

        prompt = row1["prompt"]
        answer_a = row1["response"]
        answer_b = row2["response"]

        eval_prompt = evaluation_prmp(prompt, answer_a, answer_b)
        best_answer = call_scout_model(eval_prompt)["better_answer"]
        has_hallucination = call_scout_model(eval_prompt)["hallucinated_answer"]
        reason = call_scout_model(eval_prompt)["reason"]

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
    print(f"Evaluation complete. Saved to {output_path}")

if __name__ == "__main__":
    run_evaluation("output1.csv", "output2.csv", "scout_comparison_results.csv")
