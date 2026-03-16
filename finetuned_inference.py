import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from configs.load_config import load_config
from Prompts.build_sft import chatml_format

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, 'configs/config.yaml')

CONFIG = load_config(config_path)

tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"], trust_remote_code=True)

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

adapter_path = os.path.join(BASE_DIR, CONFIG["output_dir"])
print(f"Loading LoRA adapter from: {adapter_path}")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
print("Fine-tuned model loaded.")

# Test prompts — same as baseline_inference.py for easy comparison
test_prompts = [
    {
        "title": "Strategic Question",
        "prompt": "What type of document is listed to provide a clear view of the company's organizational structure?"
    },
    {
        "title": "Hard question",
        "prompt": "Who is appointed as an attorney-in-fact and agent?"
    },
]

print("\n" + "="*60)
print("FINETUNED MODEL OUTPUTS (POST-FINETUNE)")
print("="*60)

if torch.cuda.is_available():
    vram_before = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM before generation: {vram_before:.2f} GB\n")

for i, test in enumerate(test_prompts, 1):
    formatted_prompt = chatml_format(test["prompt"])

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            temperature=CONFIG["temperature"] if CONFIG["temperature"] > 0 else None,
            do_sample=CONFIG["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start_time

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

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
