import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer
from configs.load_config import load_config
from Dataset.dataset_loader import load_qna_dataset,map_dataset_fields
from Prompts.build_sft import build_sft_prompt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, 'configs/config.yaml')

CONFIG = load_config(config_path)
tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"], trust_remote_code=True)
effective_batch_size = CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"]
# Load dataset
dataset = load_qna_dataset(
    CONFIG["dataset_name"],
    CONFIG["dataset_split"],
    CONFIG["dataset_subsample"]
)

# Map fields
dataset = dataset.map(map_dataset_fields)

# Drop rows with missing instruction or output

dataset = dataset.filter(lambda x: x["instruction"] is not None and x["output"] is not None)
dataset = dataset.map(build_sft_prompt)

# Split into train/validation
split_dataset = dataset.train_test_split(
    train_size=CONFIG["train_val_split"],
)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]


# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=CONFIG["load_in_4bit"],
    bnb_4bit_compute_dtype=CONFIG["bnb_4bit_compute_dtype"],
    bnb_4bit_quant_type=CONFIG["bnb_4bit_quant_type"],
    bnb_4bit_use_double_quant=CONFIG["bnb_4bit_use_double_quant"],
)

print(f"Loading base model: {CONFIG['base_model']}...")
base_model = AutoModelForCausalLM.from_pretrained(
    CONFIG["base_model"],
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare model for k-bit training
print("Preparing model for k-bit training...")
base_model = prepare_model_for_kbit_training(base_model)


# LoRA config
lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    target_modules=CONFIG["lora_target_modules"],
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA
print("Applying LoRA adapters...")
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Training arguments
training_args = SFTConfig(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_train_epochs"],
    max_steps=CONFIG["max_steps"],
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    per_device_eval_batch_size=CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=float(CONFIG["learning_rate"]),
    warmup_ratio=CONFIG["warmup_ratio"],
    logging_steps=CONFIG["logging_steps"],
    save_steps=CONFIG["save_steps"],
    eval_steps=CONFIG["eval_steps"],
    save_total_limit=CONFIG["save_total_limit"],
    bf16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    report_to="none",
    max_seq_length=CONFIG["max_length"],
    packing=False,
    dataset_text_field="text",
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)
print(f"Effective batch size: {effective_batch_size}")
print(f"Training steps: {CONFIG['max_steps']}")
print(f"Total samples: {CONFIG['max_steps'] * effective_batch_size}")
print(f"Optimizer: paged_adamw_8bit")
print(f"Learning rate: {CONFIG['learning_rate']}")
print(f"LoRA rank: {CONFIG['lora_r']}")
print("="*60)

# Train
print("\nStarting training...\n")
train_result = trainer.train()

# To resume from checkpoint, uncomment:
# trainer.train(resume_from_checkpoint=True)

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Total training time: {train_result.metrics.get('train_runtime', 0):.2f}s")
print(f"Samples per second: {train_result.metrics.get('train_samples_per_second', 0):.2f}")
print(f"Steps per second: {train_result.metrics.get('train_steps_per_second', 0):.4f}")

# Estimate tokens/sec
total_tokens = CONFIG['max_steps'] * effective_batch_size * CONFIG['max_length']
tokens_per_sec = total_tokens / train_result.metrics.get('train_runtime', 1)
print(f"Approximate tokens/second: {tokens_per_sec:.1f}")
print("="*60)

# Save final adapter
print(f"\nSaving adapter to {CONFIG['output_dir']}...")
trainer.save_model(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])
print("Adapter saved.")