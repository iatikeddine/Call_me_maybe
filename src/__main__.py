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

#encoded prompt ready to be interpreted by the llm
prompt_ids = llm.encode(prompts[0].prompt_value).tolist()[0]

context = prompt_ids
generated_name_ids = []

while True:
    logits = llm.get_logits_from_input_ids(context + generated_name_ids)
    
    position = len(generated_name_ids)
    
    alive = {
        name: tokens for name, tokens in fn_names_encoded.items()
        if tokens[:position] == generated_name_ids
    }
    
    valid_token_ids = list({tokens[position] for tokens in alive.values()})
    
    best_token = max(valid_token_ids, key=lambda tid: logits[tid])
    generated_name_ids.append(best_token)
    
    if generated_name_ids in fn_names_encoded.values():
        break

selected_function = llm.decode(generated_name_ids)
print(selected_function)