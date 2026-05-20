from llm_sdk import Small_LLM_Model
from parsing import functions_list, prompts
import json
import os

# 1. Initialize Model
print("Loading model...")
model = Small_LLM_Model()

# 2. Load Vocabulary Mapping for easy lookup (No transformers import needed)
print("Loading vocabulary...")
with open(model.get_path_to_tokenizer_file(), 'r') as f:
    tokenizer_data = json.load(f)
    # The vocab is usually in ['model']['vocab'] or just ['vocab']
    # Qwen uses the modern 'tokenizer.json' format
    vocab = tokenizer_data.get('model', {}).get('vocab', {})
    if not vocab:
        # fallback for some formats
        vocab = tokenizer_data.get('vocab', {})

# Helper to find ID from string
def get_id(s):
    # This is a bit simplified; real tokenizers handle multiple IDs
    # But for single characters like { or ", it works.
    ids = model.encode(s)[0].tolist()
    return ids

# 3. Constrained Generation Core
def get_best_id_constrained(current_ids, allowed_ids=None):
    logits = model.get_logits_from_input_ids(current_ids)
    
    if allowed_ids:
        # Masking: set non-allowed to very small number
        masked = [-1e10] * len(logits)
        for aid in allowed_ids:
            # Handle if allowed_id is out of range of this specific model's vocab
            if aid < len(masked):
                masked[aid] = logits[aid]
        logits = masked
    
    # Pick max
    best_val = max(logits)
    best_id = logits.index(best_val)
    return best_id

def run_function_calling():
    results = []

    for test in prompts:
        user_prompt = f"User: {test.prompt_value}\nAssistant: "
        print(f"\nProcessing: {test.prompt_value}")
        
        # Start the sequence
        # We start by forcing the initial JSON structure
        current_ids = model.encode(user_prompt)[0].tolist()
        
        # Force: {"function": "
        fixed_start = '{"function": "'
        current_ids.extend(model.encode(fixed_start)[0].tolist())
        
        # --- Step 1: Pick Function Name ---
        func_names = [f.name for f in functions_list]
        current_func_prefix = ""
        
        while True:
            # Find all possible next tokens for the remaining parts of matching functions
            possible_next_tokens = {}
            for name in func_names:
                if name.startswith(current_func_prefix) and name != current_func_prefix:
                    # What is the next token if we are at 'current_func_prefix'?
                    remaining = name[len(current_func_prefix):]
                    next_token_id = model.encode(remaining)[0].tolist()[0]
                    if next_token_id not in possible_next_tokens:
                        possible_next_tokens[next_token_id] = []
                    possible_next_tokens[next_token_id].append(name)
            
            if not possible_next_tokens:
                break
            
            if len(possible_next_tokens) == 1:
                # Only one way to go, just take it
                next_id = list(possible_next_tokens.keys())[0]
            else:
                # Multiple options, let the model decide among them
                next_id = get_best_id_constrained(current_ids, allowed_ids=list(possible_next_tokens.keys()))
            
            decoded = model.decode([next_id])
            current_func_prefix += decoded
            current_ids.append(next_id)
            
            # If we've matched a full name exactly and there are no other longer options, we stop
            if current_func_prefix in func_names:
                # Check if any other name starts with this (e.g. fn_add and fn_add_five)
                longer_options = [n for n in func_names if n.startswith(current_func_prefix) and n != current_func_prefix]
                if not longer_options:
                    break
        
        chosen_func_name = current_func_prefix
        print(f"Chosen Function: {chosen_func_name}")
        
        # Force: ", "parameters": {
        params_start = '", "parameters": {'
        current_ids.extend(model.encode(params_start)[0].tolist())
        
        # --- Step 2: Parameters ---
        # Find the definition of the chosen function
        func_def = next(f for f in functions_list if f.name == chosen_func_name)
        
        final_params = {}
        param_items = list(func_def.parameters.items())
        
        for i, (p_name, p_info) in enumerate(param_items):
            # Force: "name": 
            p_header = f'"{p_name}": '
            current_ids.extend(model.encode(p_header)[0].tolist())
            
            if p_info.type == "number":
                # For numbers: digits, dot, or the closing tokens for this field
                # The closing tokens are ',' (if more params) or '}' (if last param)
                stopper = "," if i < len(param_items) - 1 else "}"
                
                digit_ids = [vocab.get(str(d)) for d in range(10)] + [vocab.get(".")]
                digit_ids = [d for d in digit_ids if d is not None]
                stop_id = model.encode(stopper)[0].tolist()[0]
                
                param_val_str = ""
                # We loop and allow model to pick digits or the stopper
                for _ in range(20):
                    next_id = get_best_id_constrained(current_ids, allowed_ids=digit_ids + [stop_id])
                    if next_id == stop_id:
                        break
                    
                    decoded = model.decode([next_id])
                    param_val_str += decoded
                    current_ids.append(next_id)
                
                final_params[p_name] = param_val_str.strip()
                
            else: # string
                # Force: "
                current_ids.extend(model.encode('"')[0].tolist())
                
                # For strings, we allow anything EXCEPT the quote, until the model hits a quote
                # For a simple implementation, we'll just use a 'stop at quote' logic
                quote_id = model.encode('"')[0].tolist()[0]
                
                val_str = ""
                for _ in range(50):
                    # In a real professional implementation, we'd mask out the quote until 
                    # we want to allow it, but here we just let the model choose.
                    nid = get_best_id_constrained(current_ids) 
                    if nid == quote_id:
                        break
                    
                    decoded = model.decode([nid])
                    if '"' in decoded: # Backup check
                        break
                    val_str += decoded
                    current_ids.append(nid)
                
                final_params[p_name] = val_str.strip()
                # Force closing quote (if not already there)
                current_ids.extend(model.encode('"')[0].tolist())

            if i < len(param_items) - 1:
                # Force: , 
                current_ids.extend(model.encode(", ")[0].tolist())

        # --- Step 3: Close JSON ---
        current_ids.extend(model.encode("}}")[0].tolist())
        
        # Final output
        final_json = {
            "function": chosen_func_name,
            "parameters": final_params
        }
        print(f"Result: {final_json}")
        
        results.append({
            "prompt": test.prompt_value,
            "output": final_json
        })

    # Save to file
    output_path = "data/output/results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {output_path}")

if __name__ == "__main__":
    run_function_calling()
