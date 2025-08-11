import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

class LLM:
    def __init__(self, device=None):
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # --- GPU/CPU Status Print ---
        print("=" * 60)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ Using GPU: {gpu_name} ({total_vram:.1f} GB VRAM)")
        else:
            print("⚠️ CUDA not available — running on CPU (may be slow)")
        print("=" * 60)

        # --- Load tokenizer and model ---
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None
        )

    def call(self, prompt: str, max_new_tokens: int = 256, temperature=0.0):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
