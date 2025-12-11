"""
validation.py
Path: AutoCode/FineTuning/CodeLlama7B/validation.py

Description:
    Evaluates CodeLlama-7B on the TACO Eval dataset.

    Metrics Added:
    1. JSON Validity %: Can it be parsed?
    2. Schema Compliance %: Does it have 'input'/'output' keys?
    3. BLEU Score: How closely does the test case content match the ground truth? (Precision)
    4. ROUGE-L: Captures structure and completeness of the test data. (Recall)
"""

import os
import json
import torch
import evaluate # pip install evaluate rouge_score
from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../LoRAModel/CodeLlama/Output"))
DATA_FILE = os.path.join(CURRENT_DIR, "CodeLlama-TACO-Eval.jsonl")

def extract_json(text):
    """Attempt to extract JSON list from text."""
    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end != -1:
            return text[start:end]
        return text
    except:
        return text

def normalize_test_case(test_case_list):
    """Flattens a list of inputs/outputs into a single string for text comparison."""
    if not isinstance(test_case_list, list):
        return ""
    try:
        # Convert objects to string representation: "Input: X, Output: Y | Input: A..."
        return " | ".join([f"In: {item.get('input', '')} Out: {item.get('output', '')}" for item in test_case_list])
    except:
        return ""

def evaluate_model():
    print(f"--- Evaluating CodeLlama Test Generator (Syntax + Semantics) ---")

    # 1. Load Metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    # 2. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # 3. Load Data
    if not os.path.exists(DATA_FILE):
        print("Eval dataset not found. Run preprocess.py first.")
        return
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    # Stats
    valid_json = 0
    valid_schema = 0
    total = 0

    # storage for text metrics
    predictions = []
    references = []

    # 4. Evaluation Loop
    # Slicing first 50 for demo speed; remove slice for full eval
    print(f"Generating responses for {min(50, len(dataset))} samples...")
    for row in tqdm(dataset.select(range(min(50, len(dataset))))):
        full_text = row['text']

        # Split prompt and ground truth
        split_marker = "### Test Cases (JSON):"
        if split_marker not in full_text: continue

        parts = full_text.split(split_marker)
        prompt_part = parts[0] + split_marker + "\n"
        ground_truth_str = parts[1].replace("<|endoftext|>", "").strip()

        # Generate
        inputs = tokenizer([prompt_part], return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        response = generated_text[len(prompt_part):].strip()

        # A. Syntax Checks
        extracted = extract_json(response)
        is_json = False
        parsed_pred = None

        try:
            parsed_pred = json.loads(extracted)
            is_json = True
            valid_json += 1

            # Check keys
            if isinstance(parsed_pred, list) and len(parsed_pred) > 0:
                if "input" in parsed_pred[0] and "output" in parsed_pred[0]:
                    valid_schema += 1
        except:
            parsed_pred = [] # Empty list for failed parse

        # B. Prepare for Semantic Checks (Text Similarity)
        # Parse ground truth
        try:
            parsed_ref = json.loads(ground_truth_str)
        except:
            parsed_ref = []

        # Normalize to strings for BLEU/ROUGE
        predictions.append(normalize_test_case(parsed_pred))
        references.append(normalize_test_case(parsed_ref))

        total += 1

    # 5. Compute Text Metrics
    print("\nCalculating textual similarity...")
    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)

    # 6. Final Report
    print("\n" + "="*30)
    print("      EVALUATION RESULTS      ")
    print("="*30)
    print(f"Total Samples:   {total}")
    print(f"JSON Validity:   {valid_json/total*100:.2f}%")
    print(f"Schema Match:    {valid_schema/total*100:.2f}%")
    print("-" * 30)
    print(f"BLEU Score:      {bleu_score['bleu']:.4f}  (Higher is better)")
    print(f"ROUGE-L Score:   {rouge_score['rougeL']:.4f} (Structural similarity)")
    print("="*30)

if __name__ == "__main__":
    evaluate_model()