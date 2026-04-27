from llm_sdk import Small_LLM_Model
import json

llm = Small_LLM_Model()
vocab_file = llm.get_path_to_vocab_file()

with open(vocab_file, 'r') as f:
    vocab_content = json.load(f)

vocab_content_tuned = {
    key.replace("Ġ", " "): value
    for key, value in vocab_content.items()
    }

id_to_token_vocab = {
    value: key
    for key, value in vocab_content_tuned.items()}

left_brace_id = vocab_content_tuned["{"]
right_brace_id = vocab_content_tuned["}"]
quote_id = vocab_content_tuned['"']
colone_id = vocab_content_tuned[":"]
comma_id = vocab_content_tuned[","]
prompt_id = vocab_content_tuned["prompt"]
function_name_key = vocab_content_tuned["name"]
funtion_description_id = vocab_content_tuned["description"]
function_parameters_key = vocab_content_tuned["parameters"]

print(function_name_key, funtion_description_id, function_parameters_key)