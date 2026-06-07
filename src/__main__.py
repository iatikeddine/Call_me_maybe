from llm_sdk import Small_LLM_Model
import json
from parsing import functions_list, prompts

llm = Small_LLM_Model()
vocab_file = llm.get_path_to_vocab_file()

with open(vocab_file, 'r') as f:
    vocab_content = json.load(f)

reversed_vocab = {value:key for key, value in vocab_content.items()}

fn_names = []
for function in functions_list:
    fn_names.append(function.name)

#contains function name: list of tokens
fn_names_encoded = {name: llm.encode(name).tolist()[0] for name in fn_names}

def get_fn_name(prompt_ids: list) -> str:
    generated_name_ids = []
    while True:
        logits = llm.get_logits_from_input_ids(prompt_ids + generated_name_ids)
        position = len(generated_name_ids)
        alive = {
            name: tokens for name, tokens in fn_names_encoded.items()
            if tokens[:position] == generated_name_ids
        }
        valid_token_ids = list({tokens[position] for tokens in alive.values()})
        best_token = max(valid_token_ids, key=lambda x: logits[x])
        generated_name_ids.append(best_token)
        for name, tokens in fn_names_encoded.items():
            if generated_name_ids == tokens:
                return name

def is_valid_number_continuation(partial: str, token_str: str) -> bool:
    candidate = partial + token_str.strip()
    if candidate == "" or candidate == "-":
        return True
    if candidate == ".":
        return False
    try:
        float(candidate)
        return True
    except ValueError:
        # allow partial like "3." or "-3."
        return candidate.endswith(".") and candidate.count(".") == 1


def is_valid_string_continuation(partial: str, token_str: str) -> bool:
    # string must start with " and end with "
    if partial == "":
        return token_str.startswith('"')
    if partial.startswith('"') and not partial[1:].endswith('"'):
        return True
    return False


def is_complete_value(value: str, param_type: str) -> bool:
    if param_type == "number":
        try:
            float(value)
            return True
        except ValueError:
            return False
    elif param_type == "string":
        return (
            value.startswith('"') 
            and len(value) > 1 
            and value.endswith('"')
        )
    return False

def get_fn_params(prompt_ids: list, selected_function: str) -> dict:
    function = next(f for f in functions_list if f.name == selected_function)
    params = {}
    
    generated_so_far = []
    
    for param_name, param_info in function.parameters.items():
        generated_value_ids = []
        
        while True:
            logits = llm.get_logits_from_input_ids(
                prompt_ids + generated_so_far + generated_value_ids
            )
            
            if param_info.type == "number":
                valid_token_ids = [
                    tid for tid, token_str in reversed_vocab.items()
                    if is_valid_number_continuation(
                        llm.decode(generated_value_ids), token_str
                    )
                ]
            elif param_info.type == "string":
                valid_token_ids = [
                    tid for tid, token_str in reversed_vocab.items()
                    if is_valid_string_continuation(
                        llm.decode(generated_value_ids), token_str
                    )
                ]
            
            best_token = max(valid_token_ids, key=lambda x: logits[x])
            generated_value_ids.append(best_token)
            
            # check if value is complete
            current_value = llm.decode(generated_value_ids)
            if is_complete_value(current_value, param_info.type):
                params[param_name] = current_value
                generated_so_far += generated_value_ids
                break
    
    return params

if __name__ == "__main__":
    for prompt in prompts:
        print(prompt.prompt_value)
        prompt_ids = llm.encode(prompt.prompt_value).tolist()[0]
        name = get_fn_name(prompt_ids)
        params = get_fn_params(prompt_ids, name)
        print(name)
        print(params)