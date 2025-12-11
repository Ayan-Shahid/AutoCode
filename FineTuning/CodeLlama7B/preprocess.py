"""
preprocess.py
Path: AutoCode/FineTuning/CodeLlama7B/preprocess.py

Description:
    Preprocesses the BAAI/TACO dataset for CodeLlama-7B.
    - Parses input/output pairs.
    - Formats prompt for JSON Test Generation.
    - Splits data into Train (95%) and Eval (5%) sets.
"""

import os
import json
import logging
import random
from datasets import load_dataset

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATASET_ID = "BAAI/TACO"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_TRAIN_FILE = os.path.join(CURRENT_DIR, "CodeLlama-TACO-Train.jsonl")
OUTPUT_EVAL_FILE = os.path.join(CURRENT_DIR, "CodeLlama-TACO-Eval.jsonl")

# Split Ratio (0.05 = 5% for evaluation)
EVAL_RATIO = 0.05


def format_io_to_json(io_string):
    """
    Parses TACO's 'input_output' string and converts it to a clean JSON structure.
    Returns None if the data is malformed.
    """
    try:
        data = json.loads(io_string)
        inputs = data.get('inputs', [])
        outputs = data.get('outputs', [])

        if not inputs or not outputs:
            return None

        # Pair them up
        test_cases = []
        # Limit to 5 test cases to keep context window manageable
        for inp, out in zip(inputs[:5], outputs[:5]):

            # Simple sanitization
            val_in = str(inp).strip()
            val_out = str(out).strip()

            # Skip massive blocks of text/binary data
            if len(val_in) > 2000 or len(val_out) > 2000:
                continue

            test_cases.append({
                "input": val_in,
                "output": val_out
            })

        if not test_cases:
            return None

        return json.dumps(test_cases, indent=2)
    except Exception as e:
        return None


def process():
    logger.info(f"🌊 Downloading {DATASET_ID}...")
    try:
        dataset = load_dataset(DATASET_ID, split="train")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    processed_data = []

    logger.info("⚙️ Processing rows...")

    for row in dataset:
        question = row.get('question_content', row.get('question', '')).strip()
        if not question:
            continue

        io_string = row.get('input_output', '')
        if not io_string:
            continue

        target_json = format_io_to_json(io_string)
        if not target_json:
            continue

        # Construct the Full Prompt text (Prompt + Completion + EOS)
        # This formatting is what the model actually reads during training.
        full_text = f"""Below is a programming problem. You are a QA Engineer agent.
Your task is to generate unit tests in strict JSON format.

### Rules:
1. **Schema**: Output a JSON list of objects: [{{"input": <value>, "output": <value>}}]
2. **Formatting**: Do NOT wrap numbers or arrays in quotes (unless they are strings).
3. **Output**: Do not output any conversational text. Output ONLY the JSON array.

### Problem Description:
{question}

### Test Cases (JSON):
{target_json}<|endoftext|>"""

        processed_data.append({"text": full_text})

    logger.info(f"Total valid examples processed: {len(processed_data)}")

    # Shuffle and Split
    random.shuffle(processed_data)
    split_index = int(len(processed_data) * (1 - EVAL_RATIO))

    train_data = processed_data[:split_index]
    eval_data = processed_data[split_index:]

    # Save Train Data
    logger.info(f"✅ Saving {len(train_data)} training examples to {OUTPUT_TRAIN_FILE}")
    with open(OUTPUT_TRAIN_FILE, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")

    # Save Eval Data
    logger.info(f"✅ Saving {len(eval_data)} evaluation examples to {OUTPUT_EVAL_FILE}")
    with open(OUTPUT_EVAL_FILE, 'w', encoding='utf-8') as f:
        for entry in eval_data:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    process()