import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from configs.load_config import load_config
from Dataset.dataset_loader import load_qna_dataset,map_dataset_fields
from Prompts.build_sft import build_sft_prompt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, 'configs/config.yaml')

CONFIG = load_config(config_path)

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

# Split into train/validation
split_dataset = dataset.train_test_split(
    train_size=CONFIG["train_val_split"],
)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]




# Load tokenizer for diagnostics
print(f"Loading tokenizer: {CONFIG['base_model']}")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"], trust_remote_code=True)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Sample up to 500 examples for diagnostics
sample_size = min(500, len(train_dataset))
sample_dataset = train_dataset.select(range(sample_size))

# Compute token lengths
token_lengths = []
for example in sample_dataset:
    # Concatenate instruction + input + output
    text = f"{example['instruction']} {example['input']} {example['output']}"
    tokens = tokenizer(text, add_special_tokens=True)
    token_lengths.append(len(tokens["input_ids"]))

token_lengths = np.array(token_lengths)

# Map datasets to text format
train_dataset = train_dataset.map(build_sft_prompt)
val_dataset = val_dataset.map(build_sft_prompt)

print("✅ Prompts built for SFT")
print("\n📝 Sample formatted prompt:")
print(train_dataset[0]["text"][:500] + "...")