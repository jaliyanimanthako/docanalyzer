def chatml_format(user_text, system_text="You are a helpful assistant.", assistant_text=None):
    """Format text in ChatML style."""
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    if assistant_text:
        messages.append({"role": "assistant", "content": assistant_text})

    # Format as ChatML
    formatted = ""
    for msg in messages:
        formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"

    return formatted


def build_sft_prompt(row):
    """Build SFT prompt from dataset row."""
    user_text = row["instruction"]
    if row["input"]:
        user_text += f"\n\nInput: {row['input']}"

    prompt = chatml_format(
        user_text=user_text,
        system_text="You are a helpful assistant.",
        assistant_text=row["output"]
    )

    return {"text": prompt}


