"""
train.py
Path: AutoCode/FineTuning/QwenCoder7B/train.py

Description:
    Fine-tunes Qwen2.5-Coder-7B-Instruct using Unsloth.
    Uses ChatML format for instruction tuning.
"""

import os
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# --- Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(CURRENT_DIR, "QwenCoder-CCP-Train.jsonl")
EVAL_FILE = os.path.join(CURRENT_DIR, "QwenCoder-CCP-Eval.jsonl")

OUTPUT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../LoRAModel/QwenCoder/Output"))
LOG_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../LoRAModel/QwenCoder/Logs"))

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"


def train():
    print(f"--- Starting Training Pipeline for {MODEL_NAME} ---")

    # 1. Load Base Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # 2. Apply Chat Template (Crucial for Qwen/Instruct models)
    # We map the generic 'user/assistant' roles to Qwen's specific tokens if needed,
    # or use the standard 'chatml' template which Unsloth supports natively.
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )

    # 3. Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 4. Load and Format Dataset
    print(f"Loading dataset from {DATA_FILE}...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    eval_dataset = load_dataset("json", data_files=EVAL_FILE, split="train")

    # We need to apply the tokenizer's chat template to the dataset *before* training
    # or let SFTTrainer handle it. Unsloth recommends applying it to create a 'text' column.
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    eval_dataset = dataset.map(formatting_prompts_func, batched=True)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,  # Demo steps
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        logging_dir=LOG_DIR,
        report_to="tensorboard",
        save_strategy="steps",
        eval_strategy="steps",
    )

    # 6. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        args=training_args,
    )

    # 7. Train and Save
    print("Training...")
    trainer.train()

    print(f"Saving to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    train()