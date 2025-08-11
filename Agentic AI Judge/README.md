# LLM-AGENTIC-AI

LLM-AGENTIC-AI is a project designed to leverage Large Language Models (LLMs) for agentic AI tasks. This repository provides tools and scripts to run agentic workflows using LLMs on single entries or entire datasets.

## Features

- Agentic AI workflows powered by LLMs
- Process single data entries or batch datasets
- Extensible and customizable

## Requirements

- Python 3.8+

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/nnog/llm-translation-judge-EN-TL.git
   cd Agentic AI Judge
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Change directory in agentic_judge_main.py
   ```
   PROJECT_DIR = "INSERT/DIRECTORY/HERE"  # Change this to your project directory
   ```

## Usage

### Run on a Single Entry

To process a single entry, use:

```
python prompt_engineered_judge_main.py "SOURCE_TEXT" "TRANSLATION"
```

### Run on an Entire Dataset

To process an entire dataset, use:

```
python prompt_engineered_judge_main.py
```
