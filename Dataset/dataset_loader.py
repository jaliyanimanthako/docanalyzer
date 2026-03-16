import os
import sys
from datasets import load_dataset, Dataset
import json
from configs.load_config import load_config

# Add the parent directory (project root) to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, '../configs/config.yaml')

CONFIG = load_config(config_path)

def load_qna_dataset(dataset_name, split, subsample, seed=42):
    """Load dataset with robust field mapping and fallback."""

    try:
        # Try loading from Hugging Face
        print(f"📥 Loading dataset: {dataset_name}...")
        dataset = load_dataset(dataset_name, split=split)
        dataset = dataset.shuffle(seed=seed).select(range(min(subsample, len(dataset))))
        print(f"✅ Loaded {len(dataset)} examples from Hugging Face")

    except Exception as e:
        print(f"⚠️ Failed to load from Hugging Face: {e}")
        print("🔄 Creating synthetic fallback dataset...")


    return dataset


def map_dataset_fields(example):
    """Robustly map dataset fields to instruction/input/output schema."""

    # Try to find instruction
    question = None
    for key in ["instruction", "question", "prompt", "task"]:
        if key in example and example[key]:
            instruction = str(example[key]).strip()
            break

    # Try to find input (optional)
    input_text = ""
    for key in ["input", "context", "passage", "history"]:
        if key in example and example[key]:
            input_text = str(example[key]).strip()
            break

    # Try to find output/target
    output = None
    for key in ["output", "response", "answer", "target", "completion"]:
        if key in example and example[key]:
            output = str(example[key]).strip()
            break

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output
    }


# Load dataset
dataset = load_qna_dataset(
    CONFIG["dataset_name"],
    CONFIG["dataset_split"],
    CONFIG["dataset_subsample"]
)

print(f"\n📊 Dataset before cleaning: {len(dataset)} examples")

# Map fields
dataset = dataset.map(map_dataset_fields)

# Drop rows with missing instruction or output
dataset = dataset.filter(lambda x: x["instruction"] is not None and x["output"] is not None)

print(f"📊 Dataset after cleaning: {len(dataset)} examples")
print(f"✅ Dropped {CONFIG['dataset_subsample'] - len(dataset)} examples with missing data\n")

# Split into train/validation
split_dataset = dataset.train_test_split(
    train_size=CONFIG["train_val_split"],
)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

print(f"📊 Train: {len(train_dataset)} | Validation: {len(val_dataset)}")
print("\n📝 Sample example:")
print(train_dataset[0])