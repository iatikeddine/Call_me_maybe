from llm_sdk import Small_LLM_Model
import json
from parsing import functions_list, prompts

llm = Small_LLM_Model()
vocab_file = llm.get_path_to_vocab_file()

with open(vocab_file, 'r') as f:
    vocab_content = json.load(f)

reversed_vocab = {value:key for key, value in vocab_content.items()}

encoded_prompt = llm.encode(prompts[0].prompt)
generated_tokens = []
identation = 0

while True:
    logits = llm.get_logits_from_input_ids(encoded_prompt.tolist() + generated_tokens)
    valid_tokens = get_valid_tokens()
