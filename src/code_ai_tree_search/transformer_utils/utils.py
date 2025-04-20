import torch
import transformers

def get_model_by_name(model_name, device):
    # ↓ load the tokenizer for *exactly* the same model you’re going to load
    print("Loading:", model_name, device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        pad_token_id=tokenizer.eos_token_id
    )
    model.to(device)

    if device.type == 'cuda' and hasattr(model, 'parallelize'):
        print("Going to parallelize")
        model.parallelize()

    return model, tokenizer
