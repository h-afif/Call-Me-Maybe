from llm_sdk import Small_LLM_Model
import json


def extract_complete_json(text: str):
    start = text.find("{")

    if start == - 1:
        return None
    
    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        if text[i] == "}":
            brace_count -= 1
        
        if brace_count == 0:
            return text[start:i+1]
        
    return None



def get_best_valid_token(logits, valid_id):
    return max(valid_id, key=lambda i: logits[i])


def build_json_valid_ids(vocab: dict):
    json_safe = set(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789*_.,:-+/\'!?()[]{}"ĠĊ'
    )
    valid = set()
    for token_str, token_id in vocab.items():
        if token_str and all(c in json_safe for c in token_str):
            valid.add(token_id)
    return valid


def load_vocabulary(model: Small_LLM_Model):
    vocab_path = model.get_path_to_tokenizer_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
    raw_vocab = tok_data.get("model", {}).get("vocab", {})
    return raw_vocab


def build_system_prompt(functions):
    lines = [
        "STRICT SYSTEM RULE: Use ONLY a matching function from the list below.",
        "If NO function matches the user's intent (even if types match), set name: \"none\".",
        "Never use an unrelated function for a different task.",
        "",
        "Available functions:",
    ]
    for fn in functions:
        params = ", ".join(
            f"{name}: {info.type}"
            for name, info in fn.parameters.items()
        )
        lines.append(f"  -{fn.name}({params}): {fn.description}")

    lines.append('\nOutput ONLY valid JSON: {"name": "<fn>", "args": {<args>}}')

    return "\n".join(lines)
