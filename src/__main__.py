import argparse
from src.json_loader import load_function_definition, load_prompts
from src.constrained_decoding import build_system_prompt
from llm_sdk import Small_LLM_Model
from src.constrained_decoding import load_vocabulary, build_json_valid_ids, get_best_valid_token, extract_complete_json
import time
import json
import os


def parse_args():
    parse = argparse.ArgumentParser(
        description="Translate natural languge prompts into function calls..."
    )

    parse.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="input file for prompts"
    )

    parse.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json"
    )

    parse.add_argument(
        "--output",
        type=str,
        default="data/output/functions_results.json"
    )

    parse.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B"
    )

    return parse.parse_args()



def main():
    print("🚀 Starting function and prompts...")
    args = parse_args()
    print("📂 Loading functions and prompts")
    functions = load_function_definition(args.functions_definition)
    if not functions:
        raise RuntimeError(
            f"No function definition found. Please provide at least one."
        )
    prompts = load_prompts(args.input)
    if not prompts:
        raise RuntimeError(
            f"No prompts found in input file. Please provide at least one."
        )
    
    print("⚙️  building system prompt")
    system = build_system_prompt(functions)
    
    print(f"🤖 Loading model: {args.model}")
    try:
        model = Small_LLM_Model(model_name=args.model)
    except OSError:
        raise RuntimeError(
            f"Model {args.model} not found or failed to download"
        )
    
    print("🔤 Building valid token IDs...")
    vocab = load_vocabulary(model)
    valid_ids = build_json_valid_ids(vocab)

    all_results = []
    start_time = time.time()

    print("⚙️  Processing prompts...\n")
    for p in prompts:
        prompt = p.prompt
        print(f"⏳ Processing promtp: {prompt}")
        full_prompt = f"{system}\n\nUser prompt: {prompt}\nAssistant:"
        input_ids = model.encode(full_prompt)
        generated_ids = input_ids[0].tolist()

        all_generated = []

        clean_json = None
        all_generated.extend(model.encode('{"name": "')[0].tolist())
        for _ in range(50):
            logits = model.get_logits_from_input_ids(generated_ids + all_generated)
            next_id = get_best_valid_token(logits, valid_ids)
            all_generated.append(next_id)
            
            text = model.decode(all_generated)
            print(text)
            clean_json = extract_complete_json(text)
            if clean_json:
                try:
                    parsed = json.loads(clean_json)
                    break
                except Exception:
                    pass

        
        if not clean_json:
            parsed = {"name": "none", "args": {}}

        all_results.append({
            "prompt": prompt,
            "name": parsed.get("name", "none"),
            "args": parsed.get("args", {})
        })

        if parsed.get("name", "none") != "none":
            print(f"  -> ✅ {parsed['name']}({parsed['args']})")
        else:
            print(f"  -> ❌ [ERROR] Could not generate function call.")

    total_time = time.time() - start_time
    all_parsed_result = [result for result in all_results if result["name"] != "none"]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_parsed_result, f, ensure_ascii=False, indent=2)
    print(f"💾 Result saves to: {args.output}")
    print("✨ Completed.")
    print(f"⏱️ Total time: {total_time:.2f} second")
    print(f"📈 Average per prompt: {total_time/len(prompt):.2f} seconds")
    print(f"🎯 Success rate: {len(all_parsed_result)}/{len(prompts)} ({len(all_parsed_result)/len(prompts)*100:.2}%)")





if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"User stop the programe.")
    except Exception as error:
        print(f"Error: {error}")