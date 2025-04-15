import csv
from datetime import datetime
import os

def get_log_file(test_name):
    os.makedirs("logs", exist_ok=True)
    return f"logs/arc_log_{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

LOG_FILE = None

def init_logger(test_name):
    global LOG_FILE
    LOG_FILE = get_log_file(test_name)
    with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Task", "Prompt", "ModelUsed", "Difficulty", "Confidence Score",
            "Response", "TokensUsed", "TimeTaken (s)", "Error", "Timestamp"
        ])

def log_result(task_num, prompt, model, difficulty, confidence, response, tokens_used=None, time_taken=None, error=None):
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            task_num,
            prompt,
            model,
            difficulty,
            confidence,
            response,
            tokens_used if tokens_used is not None else "-",
            time_taken if time_taken is not None else "-",
            error if error else "-",
            datetime.now().isoformat()
        ])
