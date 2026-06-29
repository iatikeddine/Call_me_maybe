from llm_sdk import Small_LLM_Model
import json
from src.parsing import functions_list, prompts
import json

def function_name_context() -> str:
    """
    Gets the context to feed the llm from functions data
    """
    context = "Available functions\n"
    for function in functions_list:
        context += function.name + ':' + function.description + '\n'
    return context


def valid_names_tokens(valid_names: list) -> set:
    """
    Takes encoded functions names for function definitions
    and returns a set of valid tokens to remove duplicates
    """
    valid_tokens = set()

    for encoded_name in valid_names:
        for token in encoded_name:
            valid_tokens.add(token)

    return valid_tokens


def get_function_name(prompt: str, valid_names: list) -> list:
    """
    Uses constraint decoding to get valid function name
    based on the user prompt and context
    """
    prompt_context = function_name_context()
    prompt_helper = "predict function name for this question: "
    prompt_shema = '{"function name": "'
    encoded_prompt = llm.encode(prompt_context + prompt_helper + prompt + prompt_shema).tolist()[0]
    
    token_position = 0
    valid_names_copy = valid_names.copy()

    while(len(valid_names_copy) > 1):
        logits = llm.get_logits_from_input_ids(encoded_prompt)
        valid_tokens = valid_names_tokens(valid_names_copy)

        vocab_len = len(logits)
        for x in range(vocab_len):
            if x not in valid_tokens:
                logits[x] = -float('inf')
        
        picked_token = logits.index(max(logits))

        valid_names_copy = [
            name for name in valid_names_copy
            if len(name) > token_position
            if picked_token == name[token_position]
        ]

        encoded_prompt.append(picked_token)
        token_position += 1
    
    return valid_names_copy[0]


def valid_string(encoded_prompt: list) -> str:
    """
    implements constraint decosing to predict correct string parameter
    value for a function and returns it as a string
    """
    quote_id = vocab_content['"']
    result = ''

    logits = llm.get_logits_from_input_ids(encoded_prompt)
    vocab_len = len(logits)
    for x in range(vocab_len):
        if x != quote_id:
            logits[x] = -float('inf')
    
    picked_token = logits.index(max(logits))
    encoded_prompt.append(picked_token)
    result += llm.decode([picked_token])

    generated_tokens = 0
    while generated_tokens < 25:
        logits = llm.get_logits_from_input_ids(encoded_prompt)
        picked_token = logits.index(max(logits))
        decoded = llm.decode([picked_token])

        if '"' in decoded:
            quote_position = decoded.index('"')
            valid = decoded[:quote_position + 1]
            result += valid
            
            if valid == '"':
                encoded_prompt.append(quote_id)
            else:
                valid_ids = llm.encode(valid).tolist()
                encoded_prompt.extend(valid_ids)
            return result

        encoded_prompt.append(picked_token)
        result += llm.decode([picked_token])
        generated_tokens += 1

    result += '"'
    encoded_prompt.append(quote_id)
    return result


def valid_number(encoded_prompt: list) -> str:
    """
    implements constraint decosing to predict correct number parameter
    value for a function and returns it as a string
    """
    valid_tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.', ',', '}']
    valid_tokens_encoded = [llm.encode(token).tolist()[0][0] for token in valid_tokens]
    result = ""

    comma_id = vocab_content[',']
    bracket_id = vocab_content['}']
    generated_tokens = 0

    while generated_tokens < 25:
        logits = llm.get_logits_from_input_ids(encoded_prompt)
        for x in range(len(logits)):
            if x not in valid_tokens_encoded:
                logits[x] = -float('inf')
        
        picked_token = logits.index(max(logits))
        decoded = llm.decode([picked_token])
        if ',' in decoded or '}' in decoded:
            return result

        encoded_prompt.append(picked_token)
        result += llm.decode([picked_token])
        generated_tokens += 1
    
    return result

def valid_bool(encoded_prompt) -> int:
    valid_bool = ['0' ,'1']
    valid_tokens = [llm.encode(x).tolist()[0][0] for x in valid_bool]

    logits = llm.get_logits_from_input_ids(encoded_prompt)

    vocab_len = len(vocab_content)
    for x in range(vocab_len):
        if x not in valid_tokens:
            logits[x] = -float('inf')
    
    picked_token = logits.index(max(logits))
    encoded_prompt.append(picked_token)
    return llm.decode(picked_token)

def get_fn_params(function_name: str, functions_data: list, prompt: str) -> str:
    """
    implements constraint decoding to guide the llm to predict correct
    function name based on user prompt
    """
    params_dict = {key:value.type for key, value in functions_data[function_name].parameters.items()}
    params_len = len(params_dict)

    context = (f"function name: {function_name}\n"
               f"function purpose: {functions_data[function_name].description}\n"
               f"number of parameters: {params_len}\n"
               f"question: {prompt}\n"
               f"predict the correct function parameters in JSON format "
               )
    
    context += '{'
    result = '{'

    param_position = 0
    for name, type in params_dict.items():
        field = f'"{name}": '
        context += field
        result += field

        encoded_prompt = llm.encode(context).tolist()[0]
        if type == "string":
            value = valid_string(encoded_prompt)


        elif type in ["number", "integer"]:
            value = valid_number(encoded_prompt)
        
        elif type == "boolean":
            value = valid_bool(encoded_prompt)

        result += value
        context += value
        
        if param_position == params_len - 1:
            result += '}'
        else:
            result += ','
            context += ','
        param_position += 1

    result = result.replace("\\", "\\\\")
    return result

def cast_value(value: str, type: str) -> int | float | bool:
    if type == "number":
        return float(value)
    elif type == "integer":
        return int(value)
    elif type == "boolean":
        return bool(int(value))
    return value


if __name__ == "__main__":

    llm = Small_LLM_Model()
    vocab_file = llm.get_path_to_vocab_file()

    with open(vocab_file, 'r') as f:
        vocab_content = json.load(f)
    
    # FUNCTION NAME CONSTRAINT DECODING
    valid_names = [llm.encode(function.name).tolist()[0] for function in functions_list]
    predicted_function_names = []

    for prompt in prompts:
        result = get_function_name(prompt.prompt, valid_names)
        predicted_function_names.append(llm.decode(result))
    

    # FUNCTION PARAMS CONSTRAINT DECODING
    functions_data = {function.name: function for function in functions_list}
    predicted_functions_params = []

    for name, prompt in zip(predicted_function_names, prompts):
        predicted_functions_params.append(get_fn_params(name, functions_data, prompt.prompt))

    results = []    
    for name, params, prompt in zip(predicted_function_names, predicted_functions_params, prompts):

        params_to_dict = json.loads(params)

        if len(params_to_dict) == 1:
            (param_name, param_value), = params_to_dict.items()
            param_type = functions_data[name].parameters[param_name]
            params_to_dict[param_name] = cast_value(param_value, param_type.type)

        else:
            for param_name, param_value in params_to_dict.items():
                param_type = functions_data[name].parameters[param_name]
                params_to_dict[param_name] = cast_value(param_value, param_type.type)
                

        results.append({
            "prompt": prompt.prompt,
            "name": name,
            "parameters": params_to_dict,
        })

    with open("output.json", "w") as f:
        json.dump(results, f, indent=2)