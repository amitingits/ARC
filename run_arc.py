import json
import time
from utils.logger import init_logger, log_result
from models.mistral_local import call_mistral
from models.scout_api import call_scout_model
from utils.router import route_decider, confidence

DATA_FILE = "data/set2.jsonl"
TEST_NAME = "hybrid_arc_set2"

def load_tasks(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    tasks = load_tasks(DATA_FILE)
    init_logger(TEST_NAME)

    for i, task in enumerate(tasks, 1):
        prompt = task["prompt"]
        try:
            confidence_score = confidence(prompt)
            route_decision = route_decider(confidence_score)
            model_used = "Mistral" if route_decision == "edge" else "Llama 4 Scout"

            if route_decision == "edge":
                start_time = time.time()
                response = call_mistral(prompt)
                duration = round(time.time() - start_time, 3)
                tokens = "-"
            else:
                start_time = time.time()
                result = call_scout_model(prompt)
                duration = round(time.time() - start_time, 3)
                response = result["response"]
                tokens = result.get("tokens_used", "-")
            print(f"[Task {i}] Completed | Duration: {duration}s")
            log_result(
                task_num=i, 
                prompt=prompt, 
                model=model_used, 
                difficulty=task["difficulty"],
                confidence=confidence_score, 
                response=response, 
                tokens_used=tokens, 
                time_taken=duration,
                error=None
            )

        except Exception as e:
            log_result(  
                task_num=i,
                prompt=prompt,
                model="Unknown",
                difficulty=task["difficulty"],
                confidence="-",
                response="-",
                tokens_used="-",
                time_taken="-",
                error=str(e)
            )
    print(f"Arc test(hybrid) completed and logged under '{TEST_NAME}'")

if __name__ == "__main__":
    main()
