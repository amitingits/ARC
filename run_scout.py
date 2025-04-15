import json
import time
from models.scout_api import call_scout_model
from utils.logger import init_logger, log_result

INPUT_FILE = "data/set1.jsonl"
TEST_NAME = "scout_only_set1"
RATE_LIMIT = 20       # Number of requests before cooldown
COOLDOWN_TIME = 60

def load_dataset():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]

def main():
    dataset = load_dataset()
    init_logger(TEST_NAME)
    request_count = 0

    for i, item in enumerate(dataset, start=1):
        # Rate limiting
        if request_count >= RATE_LIMIT:
            print(f"⏸ Rate limit hit. Waiting {COOLDOWN_TIME} seconds...")
            time.sleep(COOLDOWN_TIME)
            request_count = 0
        
        prompt = item["prompt"]
        try:
            start = time.time()
            response = call_scout_model(prompt)
            duration = round(time.time() - start, 3)
            print(f"[Task {i}] Completed | Duration: {duration}s")

            log_result(
                task_num=i,
                prompt=prompt,
                model="Scout",
                difficulty=item["difficulty"],
                confidence="-",
                response=response["response"],
                tokens_used=response["tokens_used"],
                time_taken=duration,
                error=None
            )
            request_count += 1
        except Exception as e:
            log_result(
                task_num=i,
                prompt=prompt,
                model="Scout",
                difficulty=item["difficulty"],
                confidence="-",
                response="-",
                tokens_used="-",
                time_taken="-",
                error=str(e)
            )

    print(f"Scout-only test completed and logged under '{TEST_NAME}'")

if __name__ == "__main__":
    main()
