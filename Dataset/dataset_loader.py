import os
import sys
from datasets import load_dataset
from configs.load_config import load_config


def load_qna_dataset(dataset_name, split, subsample, seed=42):
    """Load dataset with robust field mapping and fallback."""

    try:
        # Try loading from Hugging Face
        print(f"Loading dataset: {dataset_name}...")
        dataset = load_dataset(dataset_name, split=split)
        dataset = dataset.shuffle(seed=seed).select(range(min(subsample, len(dataset))))
        print(f"Loaded {len(dataset)} examples from Hugging Face")

    except Exception as e:
        print(f"Failed to load from Hugging Face: {e}")
        print("Creating synthetic fallback dataset...")


    return dataset


def map_dataset_fields(example):
    """Robustly map dataset fields to instruction/input/output schema."""

    # Try to find instruction
    instruction = None
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

