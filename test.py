from llm_sdk import Small_LLM_Model
import json

model = Small_LLM_Model()
prompt = "What is the sum of 2 and 3?"

tensor_res = model.encode(prompt)
int_list = tensor_res[0].tolist()
logits_result = model.get_logits_from_input_ids(int_list)
path_to_vocab = model.get_path_to_vocab_file()

# enumerating vocab_file
the_one = max(logits_result)

token_idx = logits_result.index(the_one)
token_str = model.decode([token_idx])
print(token_idx)
print(token_str)