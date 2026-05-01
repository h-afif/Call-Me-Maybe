# 🚀 Call Me (Maybe): Constrained Function Calling for SLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/transformers-5.6.2-orange.svg)](https://github.com/huggingface/transformers)
[![Torch](https://img.shields.io/badge/torch-2.11.0-red.svg)](https://pytorch.org/)

**Call Me (Maybe)** is a lightweight framework designed to bridge the gap between natural language prompts and structured function calls using **Small Language Models (SLMs)**. By leveraging **Constrained Decoding**, it forces the model to output valid, parseable JSON even when using extremely small models (like Qwen 0.6B) that typically struggle with strict formatting.

---

## ✨ Key Features

- 🎯 **Guaranteed JSON Output**: Uses logit filtering to ensure the model only produces characters valid for JSON structures.
- ⚡ **SLM Optimized**: Specifically tuned for performance on lightweight models (0.5B - 1.5B parameters).
- 🛠️ **Dynamic Tool Definitions**: Easily define your functions and their schemas in simple JSON files.
- 🔒 **Local & Private**: Runs entirely on your machine using Hugging Face Transformers.
- 🚀 **Pre-filled Generation**: Nudges the model by pre-filling the JSON start, significantly improving consistency.

---

## ⚙️ How It Works

The core of this project lies in `src/constrained_decoding.py`. Instead of letting the model freely generate text, we intercept the prediction process:

1. **Vocabulary Filtering**: At each step, we identify which tokens in the model's vocabulary consist *only* of "JSON-safe" characters (letters, numbers, braces, quotes, etc.).
2. **Logit Manipulation**: We filter the model's output logits, allowing only the valid tokens to be selected.
3. **State Management**: The system pre-fills the beginning of the response (`{"name": "`) to guide the SLM toward the correct structure immediately.
4. **Validation**: The generation stops as soon as a complete JSON object is detected and verified.

---

## 📂 Project Structure

```text
.
├── data/
│   ├── input/                # Function definitions and test prompts
│   └── output/               # Results saved as structured JSON
├── llm_sdk/                  # Custom wrapper for local LLM inference
├── src/
│   ├── models/               # Pydantic models for data validation
│   ├── constrained_decoding.py # Core logit filtering logic
│   ├── json_loader.py        # Data loading utilities
│   └── __main__.py           # Main execution entry point
├── pyproject.toml            # Project dependencies (uv-compatible)
└── README.md                 # You are here!
```

---

## 🚀 Getting Started

### 1. Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management.

```bash
# Clone the repository
git clone <your-repo-url>
cd Call_me_for_youtube

# Sync dependencies
uv sync
```

### 2. Define Your Functions

Edit `data/input/functions_definition.json` to define the tools available to the model:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    }
  }
]
```

### 3. Run the Pipeline

Execute the main script to process your prompts:

```bash
uv run python -m src --input data/input/function_calling_tests.json --model Qwen/Qwen3-0.6B
```

---

## 📈 Performance Tracking

The system provides detailed metrics after each run:
- **Success Rate**: Percentage of prompts successfully converted to function calls.
- **Latency**: Total time and average time per prompt.
- **Accuracy**: (Requires manual verification of the output JSON).

---

## 🤖 Supported Models

While designed for the **Qwen** series (0.5B, 0.6B, 1.5B), the system is compatible with most Causal-LM models available on Hugging Face. Small models are recommended for maximum speed and lower VRAM usage.

---
*Developed with ❤️ for efficient local AI workflows.*
