from llm_sdk import Small_LLM_Model
import json
from parsing import functions_list, prompts

llm = Small_LLM_Model()
vocab_file = llm.get_path_to_vocab_file()

with open(vocab_file, 'r') as f:
    vocab_content = json.load(f)

reversed_vocab = {value:key for key, value in vocab_content.items()}

def get_context() -> str:
    context = "available functions:\n"
    for function in functions_list:
        context += function.name + " -> " + function.description + "\n"
    return context

def get_fn_name(prompt: str, valid_function_names: list) -> str:
    prompt_context = get_context()
    prompt_helper = "predict function name for this question: "
    prompt_shema = '{"function name": "'
    encoded_prompt = llm.encode(prompt_context + prompt_helper + prompt + prompt_shema).tolist()[0]

    token_position = 0
    while len(valid_function_names) > 1:
        logits = llm.get_logits_from_input_ids(encoded_prompt)
        predicted_token = logits.index(max(logits))

        tokens_in_current_position = [
            function[token_position]
            for function in valid_function_names
            if len(function) > token_position
        ]

        while predicted_token not in tokens_in_current_position:
            logits[predicted_token] = -float("inf")
            predicted_token = logits.index(max(logits))

        valid_function_names = [
            name for name in valid_function_names
            if len(name) > token_position
            if predicted_token == name[token_position]
        ]

        encoded_prompt.append(predicted_token)
        token_position += 1

    return llm.decode(valid_function_names[0])


def validate_param(value: str, expected_type: str) -> bool:
    if expected_type == "string":
        return True

    if expected_type == "number":
        try:
            float(value)
            return True
        except ValueError:
            return False
    if expected_type == "bool":
        if value.lower() == "true" or value.lower() != "false":
            return True

    return False

def get_fn_params(all_functions: dict, function_name: str, user_prompt: str) -> str:
    function = all_functions[function_name]

    generated = ""
    for param_name, param_info in function.parameters.items():

        context = (
            f"function: {function_name}\n"
            f"parameter: {param_name}\n"
            f"type: {param_info.type}\n"
            f"user request: {user_prompt}\n"
            f"predict this parameter based on user request\n"
            f'{param_name} = "'
        )

        encoded = llm.encode(context).tolist()[0]
        i = 0
        while i < 20:
            logits = llm.get_logits_from_input_ids(encoded)
            token = logits.index(max(logits))
            encoded.append(token)
            piece = llm.decode([token])
            generated += piece
            i += 1
    return generated

if __name__ == "__main__":
    function_names_encoded = [llm.encode(function.name).tolist()[0]
                                         for function in functions_list]
    function_names_results = []
    function_params_results = []
    all_functions = {
        function.name: function
        for function in functions_list
    }
    for prompt in prompts:
        function_names_results.append(get_fn_name(prompt.prompt_value, function_names_encoded))
    
    for prompt, name in zip(prompts, function_names_results):
        print(prompt.prompt_value)
        function_params_results.append(
            get_fn_params(all_functions, name, prompt.prompt_value))
        print(name)
        print(get_fn_params(all_functions, name, prompt.prompt_value))
