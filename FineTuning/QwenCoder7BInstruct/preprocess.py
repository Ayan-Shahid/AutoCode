"""
preprocess.py
Path: AutoCode/FineTuning/QwenCoder7B/preprocess.py

Description:
    Preprocesses the deepmind/code_contests dataset for Qwen2.5-Coder-7B-Instruct.
    Formats the data into the standard ChatML structure required for
    instruction fine-tuning and splits into Train/Eval sets.
"""

import os
import json
import logging
import random
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATASET_NAME = "deepmind/code_contests"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(OUTPUT_DIR, "QwenCoder-CCP-Train.jsonl")
EVAL_FILE = os.path.join(OUTPUT_DIR, "QwenCoder-CCP-Eval.jsonl")

# Standard Qwen System Prompt
SYSTEM_PROMPT = """You are an expert AI coding assistant specializing in Python.
Your task is to provide a complete, efficient, and correct Python solution 
to the user's competitive programming problem.
Ensure your code reads from standard input (stdin) and prints to standard output (stdout).
"""

def get_python_solution(solutions_data):
    """
    Attempts to extract a Python 3 solution from the dataset.
    CodeContests 'solutions' is a dictionary of lists: {'language': [], 'solution': []}
    """
    try:
        langs = solutions_data.get('language', [])
        sols = solutions_data.get('solution', [])

        # 1. Try to find Python 3 specifically (Language ID 3 is often Python in competitive platforms)
        #    Note: Dataset specific mappings vary, checking for string '3' or 'python' substring
        for idx, lang in enumerate(langs):
            # CodeContests often uses integer IDs for languages.
            # 3 is usually Python 3 in many CP datasets, but checking simply for valid solution existence first.
            if str(lang) == '3':
                return sols[idx]

        # 2. Fallback: Return the first solution if valid
        if sols and len(sols) > 0:
            return sols[0]

        return None
    except Exception:
        return None

def format_qwen_chat(example):
    """
    Formats a CodeContests example into a ChatML message list.
    """
    try:
        description = example.get('description', '').strip()
        if not description:
            return None

        # Extract Solution
        solutions_data = example.get('solutions', {})
        target_solution = get_python_solution(solutions_data)

        if not target_solution:
            return None

        # Construct the Message Hierarchy (ChatML)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description},
            {"role": "assistant", "content": target_solution}
        ]

        return {"messages": messages}

    except Exception as e:
        # logger.error(f"Error formatting row: {e}")
        return None

def main():
    logger.info(f"Loading dataset {DATASET_NAME}...")
    # Streaming to handle large dataset size without RAM explosion
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    logger.info("Starting preprocessing...")

    # Open both files
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f_train, \
         open(EVAL_FILE, 'w', encoding='utf-8') as f_eval:

        count = 0
        limit = 20000  # Adjust as needed for your compute budget
        eval_ratio = 0.05 # 5% for evaluation

        for row in dataset:
            formatted = format_qwen_chat(row)

            if formatted:
                json_line = json.dumps(formatted)

                # Random split
                if random.random() < eval_ratio:
                    f_eval.write(json_line + '\n')
                else:
                    f_train.write(json_line + '\n')

                count += 1

            if count >= limit:
                break

            if count % 1000 == 0:
                logger.info(f"Processed {count} valid examples...")

    logger.info(f"Preprocessing complete.")
    logger.info(f"Train data saved to: {TRAIN_FILE}")
    logger.info(f"Eval data saved to: {EVAL_FILE}")

if __name__ == "__main__":
    main()