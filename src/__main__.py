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
fn_names_encoded = {name: llm.encode(name).tolist() for name in fn_names}

prompt_ids = llm.encode(prompts[0].prompt).tolist()
print(prompt_ids)
