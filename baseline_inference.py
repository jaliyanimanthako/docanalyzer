import os
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import time
import torch
from configs.load_config import load_config
from transformers import AutoTokenizer
from Prompts.build_sft import chatml_format

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, 'configs/config.yaml')

CONFIG = load_config(config_path)

tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"], trust_remote_code=True)


# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=CONFIG["load_in_4bit"],
    bnb_4bit_compute_dtype=CONFIG["bnb_4bit_compute_dtype"],
    bnb_4bit_quant_type=CONFIG["bnb_4bit_quant_type"],
    bnb_4bit_use_double_quant=CONFIG["bnb_4bit_use_double_quant"],
)

print(f"📥 Loading base model: {CONFIG['base_model']}...")
base_model = AutoModelForCausalLM.from_pretrained(
    CONFIG["base_model"],
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
print("✅ Base model loaded in 4-bit")

# Test prompts
test_prompts = [
    {
        "title": "Stratergic Question",
        "prompt": "What type of document is listed to provide a clear view of the company's organizational structure?"
    },
    {
        "title": "Hard question",
        "prompt": "Who is appointed as an attorney-in-fact and agent?"
    },
]

print("\n" + "="*60)
print("BASELINE OUTPUTS (PRE-FINETUNE)")
print("="*60)

if torch.cuda.is_available():
    vram_before = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM before generation: {vram_before:.2f} GB\n")

for i, test in enumerate(test_prompts, 1):
    # Format as ChatML
    formatted_prompt = chatml_format(test["prompt"])

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(base_model.device)

    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            temperature=CONFIG["temperature"] if CONFIG["temperature"] > 0 else None,
            do_sample=CONFIG["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    elapsed = time.time() - start_time

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant response
    if "<|im_start|>assistant" in generated_text:
        response = generated_text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    else:
        response = generated_text[len(formatted_prompt):].strip()

    tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

    print(f"\n[{i}] {test['title']}")
    print("-" * 60)
    print(f"Prompt: {test['prompt'][:100]}...")
    print(f"\nResponse:\n{response}")
    print(f"\nLatency: {elapsed:.2f}s | Tokens: {tokens_generated} | Speed: {tokens_per_sec:.1f} tok/s")
    print("-" * 60)

if torch.cuda.is_available():
    vram_after = torch.cuda.memory_allocated() / 1024**3
    print(f"\nVRAM after generation: {vram_after:.2f} GB")
    print(f"VRAM delta: {vram_after - vram_before:.2f} GB")

print("\n" + "="*60)