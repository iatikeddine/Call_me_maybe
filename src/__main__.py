from llm_sdk import Small_LLM_Model
import json
from parsing import functions_list, prompts
from enum import Enum

class States(Enum):
    START = "start"
    KEY = "key"
    VALUE = "value"
    END = "end"

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

def check_start_state(token: int) -> bool:
    bracket_token = llm.encode("{").tolist()[0][0]
    if token != bracket_token:
        return False
    return True

def check_key_state(token: int, step: int) -> bool:
    quote_token = llm.encode('"').tolist()[0][0]

    if step == 0 or step == 2:
        return token == quote_token

    return False


def get_fn_parameters(function_name: str, number_of_params: int, prompt: str) -> str:

    num_alpha = {0: "a", 1: "b", 2: "c", 3: "d"}
    result = ""
    state = States.START.value
    context = (
        f"function name: {function_name}\n"
        f"number of parameters: {number_of_params}\n"
        f"question: {prompt}\n"
        f"predict the best function parameters for this question ")
    encoded_prompt = llm.encode(context).tolist()[0]

    for x in range(number_of_params):
        if state == "start":
            logits = llm.get_logits_from_input_ids(encoded_prompt)
            chosed_token = logits.index(max(logits))
            while not check_start_state(chosed_token):
                logits[chosed_token] = -float('inf')
                chosed_token = logits.index(max(logits))
            encoded_prompt.append(chosed_token)
            result += llm.decode([chosed_token])
            state = States.KEY.value
        
        step = 0
        if state == "key":
            if step == 0:
                logits = llm.get_logits_from_input_ids(encoded_prompt)
                chosed_token = logits.index(max(logits))
                while not check_key_state(chosed_token, step):
                    logits[chosed_token] = -float("inf")
                    chosed_token = logits.index(max(logits))
                encoded_prompt.append(chosed_token)
                result += llm.decode([chosed_token])
                step += 1
            if step == 1:
                encoded_prompt.append(llm.encode(num_alpha[x]).tolist()[0][0])
                result += num_alpha[x]
                step += 1
            if step == 2:
                tokens = llm.encode('",').tolist()[0]
                encoded_prompt.extend(tokens)
                result += '": ,'
            state = States.START.value


    result += '}'

    return result


if __name__ == "__main__":
    function_names_encoded = [llm.encode(function.name).tolist()[0]
                                         for function in functions_list]
    function_names_results = []
    function_params_results = []

    for prompt in prompts:
        function_names_results.append(get_fn_name(prompt.prompt_value, function_names_encoded))

    params_len = {function.name: len(function.parameters) for function in functions_list}

    for name, prompt in zip(function_names_results, prompts):
        function_params_results.append(get_fn_parameters(name, params_len[name], prompt.prompt_value))
    
    for name, params in zip(function_names_results, function_params_results):
        print(name)
        print(params)