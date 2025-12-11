"""
train.py
Path: AutoCode/FineTuning/CodeLlama7B/train.py

Description:
    Fine-tunes CodeLlama-7B-Instruct using Unsloth/QLoRA on BAAI/TACO.
    Includes Validation/Evaluation steps.
"""

import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# --- Configuration Paths ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(CURRENT_DIR, "CodeLlama-TACO-Train.jsonl")
EVAL_FILE = os.path.join(CURRENT_DIR, "CodeLlama-TACO-Eval.jsonl")

# Output Paths (normalized)
OUTPUT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../LoRAModel/CodeLlama/Output"))
LOG_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../LoRAModel/CodeLlama/Logs"))

# --- Model Hyperparameters ---
# Using Instruct version to better follow the "QA Engineer" prompt
MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
MAX_SEQ_LENGTH = 2048
DTYPE = None  # Auto-detects based on GPU (Float16 for T4, Bfloat16 for Ampere)
LOAD_IN_4BIT = True

def train():
    print(f"--- Starting Training Pipeline for {MODEL_NAME} ---")

    # 1. Load Unsloth Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # 2. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 3. Load Datasets
    print("Loading datasets...")
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(EVAL_FILE):
        raise FileNotFoundError("Train/Eval files not found. Run preprocess.py first.")

    dataset_train = load_dataset("json", data_files=TRAIN_FILE, split="train")
    dataset_eval = load_dataset("json", data_files=EVAL_FILE, split="train") # 'split' is technically just loading the file

    print(f"Train Size: {len(dataset_train)} | Eval Size: {len(dataset_eval)}")

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100, # Set to -1 and use num_train_epochs for full run
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        logging_dir=LOG_DIR,
        report_to="tensorboard",

        # Evaluation Config
        eval_strategy="steps",
        eval_steps=20, # Evaluate every 20 steps
        save_strategy="no", # We save manually at end to save space
    )

    # 5. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        args=training_args,
    )

    # 6. Train
    print("Starting training loop...")
    trainer.train()

    # 7. Save
    print(f"Saving artifacts to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete.")

if __name__ == "__main__":
    train()