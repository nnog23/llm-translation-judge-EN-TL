# agentic_judge.py
# Converted from Colab notebook:  to pure Python script

import os
import sys
import importlib
import json
import math
import random
import statistics
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from scipy.stats import spearmanr

# 1. Environment setup - make sure dependencies are installed before running this script:
# transformers, accelerate, huggingface_hub, pyyaml, tqdm, scipy, pandas, torch

# 2. Configure your project directory here:
PROJECT_DIR = "C:/Users/Aira/Documents/GitHub/LLM-AGENTIC-AI"

# 3. Add LLM package directory to sys.path
LLM_DIR = os.path.join(PROJECT_DIR, 'LLM')
if LLM_DIR not in sys.path:
    sys.path.insert(0, LLM_DIR)

# 4. Import your modules and reload if needed
import logger
import llm_api
import planner
from memory import Memory
from agent_tools import Toolset

importlib.reload(logger)
importlib.reload(llm_api)
importlib.reload(planner)

from logger import AgentLogger
from llm_api import LLM

print("Imported LLM package components successfully.")

# 5. Check for GPU availability
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✅ Using GPU: {gpu_name} ({total_vram:.1f} GB VRAM)")
else:
    raise RuntimeError("❌ No GPU detected. Please run on a GPU-enabled environment.")

# 6. Initialize model wrapper
device = "cuda"
llm = LLM(device=device)
print("LLM object created.")

# 7. File paths
CRITERIA_PATH = os.path.join(PROJECT_DIR, "criteria.yml")
GLOSSARY_PATH = os.path.join(PROJECT_DIR, "domain_glossary.csv")
TRACE_DIR = os.path.join(PROJECT_DIR, "results", "agent_traces")
RESULTS_FILE = os.path.join(PROJECT_DIR, "results", "results_agent_judge.json")

os.makedirs(TRACE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

# 8. Initialize components
memory = Memory(criteria_path=CRITERIA_PATH, glossary_path=GLOSSARY_PATH)
tools = Toolset(glossary_path=GLOSSARY_PATH)
logger = AgentLogger(TRACE_DIR)
planner_instance = planner.Planner(memory=memory, tools=tools, logger=logger, device=device)

print("Agent components initialized.")

# 9. Load dataset
dataset_path = os.path.join(PROJECT_DIR, "LLM_dataset.csv")
dataset = []

if os.path.exists(dataset_path):
    try:
        df = pd.read_csv(dataset_path, encoding='utf-8')
        for _, row in df.iterrows():
            dataset.append({
                "source": row['Source Text (English)'],
                "translation": row['Target Text (Filipino)'],
                "human_score": row['Final Score                          (1 - lowest, 5 - highest)']
            })
        print(f"Loaded {len(dataset)} examples from dataset.csv")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        # Fallback sample data
        dataset = [
            {
                "source": "The Department of Health announced a new vaccine rollout.",
                "translation": "Inanunsyo ng Kagawaran ng Kalusugan ang bagong programa ng pagbabakuna.",
                "human_score": 5
            },
            {
                "source": "Blood pressure must be monitored regularly for patients with diabetes.",
                "translation": "Ang presyon ng dugo ay dapat regular na sinusuri para sa mga pasyenteng may diabetes.",
                "human_score": 3
            }
        ]
        print("Using fallback sample dataset:", len(dataset))
else:
    # Fallback sample data
    dataset = [
        {
            "source": "The Department of Health announced a new vaccine rollout.",
            "translation": "Inanunsyo ng Kagawaran ng Kalusugan ang bagong programa ng pagbabakuna.",
            "human_score": 5
        },
        {
            "source": "Blood pressure must be monitored regularly for patients with diabetes.",
            "translation": "Ang presyon ng dugo ay dapat regular na sinusuri para sa mga pasyenteng may diabetes.",
            "human_score": 3
        }
    ]
    print("Using fallback sample dataset:", len(dataset))

# 10. Evaluation function for DATASET
def run_and_save(dataset, planner, out_file):
    with open(out_file, 'a', encoding='utf-8') as fout:
        for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
            try:
                result = planner.evaluate(sample)
            except Exception as e:
                result = {
                    "error": str(e),
                    "criteria_scores": {},
                    "raw_sum": 0,
                    "overall_score": 1,
                    "label": "error",
                    "explanation": {}
                }
            result_meta = {
                "index": i,
                "source": sample.get("source"),
                "translation": sample.get("translation"),
                "reference": sample.get("reference", None),
                "human_score": sample.get("human_score", None),
                "result": result
            }
            fout.write(json.dumps(result_meta, ensure_ascii=False) + "\n")
    print("Saved results to", out_file)

run_and_save(dataset, planner_instance, RESULTS_FILE)

# Helper Function to evaluate single pairr
def evaluate_single_pair(source_text, translation_text, planner):
    sample = {
        "source": source_text,
        "translation": translation_text
    }
    try:
        result = planner.evaluate(sample)
    except Exception as e:
        result = {
            "error": str(e),
            "criteria_scores": {},
            "raw_sum": 0,
            "overall_score": 1,
            "label": "error",
            "explanation": {}
        }
    result_meta = {
        "source": sample["source"],
        "translation": sample["translation"],
        "result": result
    }
    print(json.dumps(result_meta, ensure_ascii=False, indent=4))
    return result_meta


# 11. Compute Spearman correlation
def compute_spearman(results_file):
    llm_scores = []
    human_scores = []

    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            human = entry.get("human_score")
            model_out = entry.get("result", {})
            overall = model_out.get("overall_score")
            if human is not None and overall is not None:
                human_scores.append(float(human))
                llm_scores.append(float(overall))

    def is_nan(x):
        try:
            return math.isnan(float(x))
        except:
            return str(x).lower() == 'nan'

    filtered_pairs = [(h, l) for h, l in zip(human_scores, llm_scores) if not (is_nan(h) or is_nan(l))]

    if len(filtered_pairs) >= 2:
        human_scores_clean, llm_scores_clean = zip(*filtered_pairs)
        rho, pval = spearmanr(human_scores_clean, llm_scores_clean)
        print(f"Spearman rho: {rho:.4f}, p-value: {pval:.4g} (N={len(human_scores_clean)})")
    else:
        print("Not enough valid data points after removing NaNs.")

compute_spearman(RESULTS_FILE)

# 12. Check for missing explanations
def check_missing_explanations(results_file):
    missing_expl = []
    expected_keys = {"accuracy", "fluency", "coherence", "cultural_appropriateness", "guideline_adherence", "completeness"}

    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            source = entry.get("source")

            if source is None:
                continue
            if isinstance(source, float) and math.isnan(source):
                continue
            if isinstance(source, str) and source.strip().lower() == "nan":
                continue

            res = entry.get("result", {})
            expl = res.get("explanation", {})
            missing = expected_keys - set(expl.keys())
            if missing:
                missing_expl.append({"index": entry.get("index"), "missing": list(missing)})

    print("Number of entries missing some criterion explanations:", len(missing_expl))
    print(missing_expl[:10])

check_missing_explanations(RESULTS_FILE)

# 13. Pretty print results (optional, for debugging)
def pretty_print_results(results_file):
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                print(json.dumps(entry, ensure_ascii=False, indent=4))
                print("-" * 20)
    except FileNotFoundError:
        print(f"Error: The file {results_file} was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

# pretty_print_results(RESULTS_FILE)  # Uncomment to run

# 14. Consistency check on a few samples
def consistency_check(dataset, planner, samples=5, repeats=5):
    def is_valid_text(text):
        if text is None:
            return False
        if isinstance(text, float) and math.isnan(text):
            return False
        if isinstance(text, str) and text.strip().lower() == "nan":
            return False
        return True

    filtered_dataset = [s for s in dataset if is_valid_text(s.get("source")) and is_valid_text(s.get("translation"))]

    K = min(samples, len(filtered_dataset))
    indices = random.sample(range(len(filtered_dataset)), K)
    consistency_report = []

    for idx in indices:
        sample = filtered_dataset[idx]
        scores = []
        for _ in tqdm(range(repeats), desc=f"Evaluating sample {idx}", leave=False):
            out = planner.evaluate(sample)
            scores.append(out.get("overall_score"))
        mean = statistics.mean(scores)
        stdev = statistics.pstdev(scores) if mean != 0 else 0.0
        pct_var = (stdev / mean) * 100 if mean != 0 else 0.0
        consistency_report.append({
            "index": idx,
            "scores": scores,
            "mean": mean,
            "stdev": stdev,
            "pct_var": pct_var
        })
    return consistency_report

consistency_report = consistency_check(dataset, planner_instance)
print(consistency_report)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Run single pair evaluation
        source_text = sys.argv[1]
        translation_text = sys.argv[2]
        evaluate_single_pair(source_text, translation_text, planner_instance)
    else:
        # Default: run full dataset
        run_and_save(dataset, planner_instance, RESULTS_FILE)
        compute_spearman(RESULTS_FILE)
        check_missing_explanations(RESULTS_FILE)
