"""
evaluate.py
Path: AutoCode/FineTuning/QwenCoder7B/evaluate.py

Description:
    Evaluates Qwen2.5-Coder solution generation with robust offline metrics.

    Metrics:
    1. BLEU Score (Precision) - Standard textual match.
    2. ROUGE-L (Recall) - Measures longest common subsequences (good for structure).
    3. Syntax Validity (Parsability) - Checks if the code is valid Python (compiles).
"""

import os
import torch
import evaluate  # pip install evaluate rouge_score
import ast
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../LoRAModel/QwenCoder/Output"))
DATA_FILE = os.path.join(CURRENT_DIR, "QwenCoder-CCP-Eval.jsonl")

def check_syntax_validity(code_str):
    """
    Checks if the generated code is valid Python syntax.
    Returns 1 if valid, 0 if syntax error.
    """
    try:
        ast.parse(code_str)
        return 1
    except SyntaxError:
        return 0
    except Exception:
        return 0

def evaluate_model():
    print(f"--- Evaluating QwenCoder Solution Agent ---")

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

    # 3. Setup Tokenizer for ChatML
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    )

    # 4. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"Dataset not found at {DATA_FILE}")
        return

    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    predictions = []
    references = []
    syntax_scores = []

    print(f"Generating solutions for {min(50, len(dataset))} samples...")

    # 5. Evaluation Loop
    for i in tqdm(range(min(50, len(dataset)))):
        row = dataset[i]
        messages = row['messages']

        # Prepare Inputs: Everything before the final Assistant response
        input_msgs = messages[:-1]
        reference_code = messages[-1]['content']

        # Format Prompt
        inputs = tokenizer.apply_chat_template(
            input_msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        # Generate
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=1024, # Code can be long
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.2, # Lower temp for coding tasks
            do_sample=True
        )

        # Decode
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # Strip prompt to get just the generated code
        prompt_decoded = tokenizer.decode(inputs[0], skip_special_tokens=True)
        # Handle potential overlap or slight decoding diffs
        generated_code = decoded.replace(prompt_decoded, "").strip()

        # Cleanup: Remove markdown code blocks if present
        if generated_code.startswith("```python"):
            generated_code = generated_code.replace("```python", "", 1)
        if generated_code.startswith("```"):
            generated_code = generated_code.replace("```", "", 1)
        if generated_code.endswith("```"):
            generated_code = generated_code[:-3]

        predictions.append(generated_code.strip())
        references.append(reference_code.strip())

        # Syntax Check
        syntax_scores.append(check_syntax_validity(generated_code))

    # 6. Calculate Metrics
    print("\nCalculating metrics...")

    # Textual Similarity
    results_bleu = bleu.compute(predictions=predictions, references=references)
    results_rouge = rouge.compute(predictions=predictions, references=references)

    # Functional Proxy (Syntax)
    avg_syntax = sum(syntax_scores) / len(syntax_scores)

    print("\n" + "="*40)
    print("      QWEN CODER EVALUATION      ")
    print("="*40)
    print(f"BLEU Score:      {results_bleu['bleu']:.4f}  (Precision)")
    print(f"ROUGE-L:         {results_rouge['rougeL']:.4f} (Recall/Structure)")
    print(f"Syntax Validity: {avg_syntax*100:.1f}%    (Parsable Python Code)")
    print("-" * 40)
    print("Sample Output snippet:")
    print(predictions[0][:300])
    print("="*40)

if __name__ == "__main__":
    evaluate_model()