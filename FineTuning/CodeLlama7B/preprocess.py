"""
preprocess.py
Path: AutoCode/FineTuning/CodeLlama7B/preprocess.py

Description:
    Preprocesses the BAAI/TACO dataset for the Test Generation model.
    The goal is to train CodeLlama-7B to take a problem description and
    output ONLY a valid JSON object containing test cases.

    Output Format:
    JSONL file where each line contains:
    {
        "text": "Full prompt + completion for training",
        "prompt": "Input part",
        "completion": "Target JSON output"
    }
"""

import os
import json
import logging
from datasets import load_dataset

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration Constants
DATASET_NAME = "BAAI/TACO"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "processed_taco_train.jsonl")

# Prompt Template
# We design the prompt to force a specific mode of operation (JSON Generator).
# The <|endoftext|> token is crucial for letting the model know when to stop generating.
PROMPT_TEMPLATE = """Below is a programming problem. You are a QA Engineer agent.
Your task is to generate a comprehensive set of unit tests in JSON format.
Each test case must strictly adhere to the schema: {{"input": <value>, "output": <value>}}.
Do not output any conversational text, only the JSON object.

### Problem Description:
{description}

### Constraints:
{constraints}

### Test Cases (JSON):
"""


def format_taco_example(example):
    """
    Transforms a single row from the TACO dataset into the training format.

    Args:
        example (dict): A row from the Hugging Face dataset.

    Returns:
        dict: Processed example containing the full text for SFT.
    """
    try:
        # 1. Extract Problem Description
        # TACO structure varies, we extract the question content.
        description = example.get('question_content', '').strip()

        # 2. Extract and Normalize Constraints
        # If explicit constraints aren't separated, we imply them or leave generic.
        # TACO often embeds them in the description.
        constraints = "See problem description for details."

        # 3. Extract Input/Output Pairs
        # TACO stores these in 'input_output' dict which contains lists.
        io_data = example.get('input_output', {})
        inputs = io_data.get('inputs', )
        outputs = io_data.get('outputs', )

        # Data Integrity Check
        if not inputs or not outputs or len(inputs) != len(outputs):
            return None

        # 4. Construct Test Cases JSON
        # We limit to 5 test cases to manage context window efficiency during training.
        test_cases = []
        limit = min(len(inputs), 5)

        for i in range(limit):
            # We must ensure the strings are properly escaped for JSON
            test_cases.append({
                "input": inputs[i],
                "output": outputs[i]
            })

        target_json = json.dumps(test_cases, indent=2)

        # 5. Construct Training Text
        # The prompt introduces the task, the target_json provides the solution.
        formatted_prompt = PROMPT_TEMPLATE.format(
            description=description,
            constraints=constraints
        )

        full_text = formatted_prompt + target_json + "\n<|endoftext|>"

        return {
            "text": full_text
        }

    except Exception as e:
        logger.warning(f"Skipping example due to error: {e}")
        return None


def main():
    logger.info(f"Loading dataset {DATASET_NAME}...")
    # Loading split="train"
    try:
        dataset = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    logger.info(f"Dataset loaded. Rows: {len(dataset)}")
    logger.info("Starting preprocessing pipeline...")

    processed_count = 0
    skipped_count = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for row in dataset:
            processed = format_taco_example(row)
            if processed:
                f.write(json.dumps(processed) + '\n')
                processed_count += 1
            else:
                skipped_count += 1

            # Log progress every 1000 items
            if (processed_count + skipped_count) % 1000 == 0:
                logger.info(f"Processed: {processed_count}, Skipped: {skipped_count}")

    logger.info(f"Preprocessing complete.")
    logger.info(f"Total Valid Examples: {processed_count}")
    logger.info(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()