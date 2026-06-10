from llm_sdk import Small_LLM_Model
import json
from parsing import functions_list, prompts

llm = Small_LLM_Model()
vocab_file = llm.get_path_to_vocab_file()

with open(vocab_file, 'r') as f:
    vocab_content = json.load(f)

reversed_vocab = {value:key for key, value in vocab_content.items()}

def get_fn_name(prompt: str, valid_function_names: list) -> str:
    prompt_helper = "predict function name for this question: "
    prompt_shema = '{"function name": "'
    encoded_prompt = llm.encode(prompt_helper + prompt + prompt_shema).tolist()[0]

    token_count = 0
    while len(valid_function_names) != 1:
        logits = llm.get_logits_from_input_ids(encoded_prompt)
        predicted_token = logits.index(max(logits))

        tokens_in_current_position = [
            function[token_count]
            for function in valid_function_names
            if len(function) > token_count
        ]

        while predicted_token not in tokens_in_current_position:
            logits[predicted_token] = -float("inf")
            predicted_token = logits.index(max(logits))

        valid_function_names = [
            name for name in valid_function_names
            if len(name) > token_count
            if predicted_token == name[token_count]
        ]

        encoded_prompt.append(predicted_token)
        token_count += 1

    return llm.decode(valid_function_names[0])

if __name__ == "__main__":
    function_names_encoded = [llm.encode(function.name).tolist()[0]
                                         for function in functions_list]
    function_names_results = []
    function_params_results = []
    for prompt in prompts:
        function_names_results.append(get_fn_name(prompt.prompt_value, function_names_encoded))
    
    for name in function_names_results:
        print(name)
