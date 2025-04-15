import json
import time
from models.mistral_local import call_mistral
from utils.logger import init_logger, log_result

INPUT_FILE = "data/set2.jsonl"
TEST_NAME = "mistral_only_set2"

def load_dataset():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]

def main():
    dataset = load_dataset()
    init_logger(TEST_NAME)

    for i, item in enumerate(dataset, start=1):
        prompt = item["prompt"]
        try:
            start = time.time()
            response = call_mistral(prompt)
            duration = round(time.time() - start, 3)
            print(f"[Task {i}] Completed | Duration: {duration}s")
            log_result(
                task_num=i,
                prompt=prompt,
                model="Mistral",
                difficulty=item["difficulty"],
                confidence="-",
                response=response,
                tokens_used="-",
                time_taken=duration,
                error=None
            )
        except Exception as e:
            log_result(
                task_num=i,
                prompt=prompt,
                model="Mistal",
                difficulty=item["difficulty"],
                confidence="-",
                response="-",
                tokens_used="-",
                time_taken="-",
                error=str(e)
            )

    print(f"Mistral-only test completed and logged under '{TEST_NAME}'")

if __name__ == "__main__":
    main()
