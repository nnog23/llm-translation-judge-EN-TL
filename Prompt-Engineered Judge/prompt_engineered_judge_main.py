# -*- coding: utf-8 -*-
"""
prompt_engineered_judge_single_pair.py
Run with:
    python prompt_engineered_judge_single_pair.py
"""

import os
import json
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURATION ---
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
FEW_SHOT_JSONL = "few_shot_examples.jsonl"  # path to optional few-shot exemplars
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0  # deterministic; set >0 for variety
REQUIRED_CRITERIA = [
    "accuracy",
    "fluency",
    "coherence",
    "cultural_appropriateness",
    "guideline_adherence",
    "completeness"
]

# ---- Few-shot loader ----
def load_few_shot_jsonl(path):
    shots = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    shots.append(json.loads(line))
                except Exception as e:
                    print("Skipping malformed JSONL line:", e)
    return shots

few_shot_shots = load_few_shot_jsonl(FEW_SHOT_JSONL)
print(f"Loaded {len(few_shot_shots)} few-shot exemplars.")

# ---- Prompt Template ----
BASE_PROMPT = r"""
You are a translation judge for English→Filipino. Evaluate the translation using SIX criteria:

1) Accuracy: (0 or 1)
2) Fluency: (0 or 1)
3) Coherence: (0 or 1)
4) Cultural Appropriateness: (0 or 1)
5) Guideline Adherence: (0 or 1)
6) Completeness: (0 or 1)

Return ONLY valid JSON in this format:
{
  "criteria": {
    "accuracy": 0|1,
    "fluency": 0|1,
    "coherence": 0|1,
    "cultural_appropriateness": 0|1,
    "guideline_adherence": 0|1,
    "completeness": 0|1
  },
  "sum_points": 0-6,
  "normalized_score": 1-5,
  "label": "excellent"|"good"|"fair"|"poor"|"very poor",
  "explanation": "<short explanation mentioning each criterion>"
}
English: "{english}"
Filipino: "{filipino}"
"""

def escape_quotes(text):
    if text is None:
        return ""
    return text.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")

def build_prompt(english, filipino, reference=None, shots=None):
    shot_texts = []
    if shots:
        for s in shots:
            exemplar_json = {
                "criteria": s.get("criteria", {k: 1 for k in REQUIRED_CRITERIA}),
            }
            exemplar_json["sum_points"] = sum(exemplar_json["criteria"].values())
            exemplar_json["normalized_score"] = (
                5 if exemplar_json["sum_points"] >= 5
                else (3 if exemplar_json["sum_points"] >= 3 else 1)
            )
            exemplar_json["label"] = (
                "excellent" if exemplar_json["normalized_score"] == 5
                else ("good" if exemplar_json["normalized_score"] == 3 else "poor")
            )
            exemplar_json["explanation"] = s.get("explanation", "")
            shot_texts.append(
                f"### Example\nEnglish: \"{escape_quotes(s['english'])}\"\nFilipino: \"{escape_quotes(s['filipino'])}\"\nDesired JSON output:\n{json.dumps(exemplar_json, ensure_ascii=False)}\n"
            )
    ref_text = f"\nReference: \"{escape_quotes(reference)}\"\n" if reference else "\n"
    prompt = BASE_PROMPT + "\n\n" + "\n".join(shot_texts) + f"\n### Now evaluate:\nEnglish: \"{escape_quotes(english)}\"\nFilipino: \"{escape_quotes(filipino)}\"{ref_text}\nReturn JSON only.\n"
    return prompt

# ---- Model Loader ----
print("Loading model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    print("Loaded model in FP16.")
except Exception as e:
    print("FP16 load failed, loading in default precision:", e)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True
    )

model.eval()
DEVICE = next(model.parameters()).device

# ---- Generate text ----
def generate_text(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, seed=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    with torch.inference_mode():
        out = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ---- JSON Extractor ----
def extract_json_from_text(text):
    try:
        return json.loads(text.strip())
    except:
        pass
    brace_stack = []
    start_idx = None
    for i, ch in enumerate(text):
        if ch == '{':
            if start_idx is None:
                start_idx = i
            brace_stack.append(ch)
        elif ch == '}' and brace_stack:
            brace_stack.pop()
            if not brace_stack:
                candidate = text[start_idx:i+1]
                try:
                    return json.loads(candidate)
                except:
                    pass
                start_idx = None
    return None

# ---- Evaluation Function ----
def evaluate_pair(english, filipino, reference=None, tries=2, temp=TEMPERATURE, seed=None, debug=False):
    prompt = build_prompt(english, filipino, reference, shots=few_shot_shots)
    last_raw = None
    for attempt in range(tries):
        raw = generate_text(prompt, temperature=temp, seed=seed)
        last_raw = raw
        parsed = extract_json_from_text(raw)
        if parsed:
            return parsed
    return {"error": "no valid JSON parsed", "raw": last_raw}

# ---- MAIN ----
if __name__ == "__main__":
    english_text = "Hello, how are you?"
    filipino_text = "Kamusta, kumusta ka?"
    reference_text = None

    seed = random.randint(1, 2**30 - 1)
    result = evaluate_pair(
        english_text,
        filipino_text,
        reference=reference_text,
        tries=2,
        temp=TEMPERATURE,
        seed=seed,
        debug=True
    )

    output_file = "single_pair_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"💾 Result saved to: {output_file}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
