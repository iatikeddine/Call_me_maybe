from llm_sdk import Small_LLM_Model
import json
from parsing import functions_list, prompts
from enum import Enum


class States(Enum):
    START = "start"
    KEY   = "key"
    VALUE = "value"
    END   = "end"


llm = Small_LLM_Model()

functions_data = {function.name: function for function in functions_list}

def get_fn_params(function_name: str, functions_data, prompt: str) -> str:
    num_of_params = len(functions_data[function_name].parameters)

    context = (f"function_name: {function_name}\n"
               f"function description: {functions_data[function_name].description}\n"
               f"number of parameters: {num_of_params}\n"
               f"question: {prompt}\n"
               f"Predict the correct function parametersin a json format\n"
               )

    context += " {"
    result = "{"

    for x, (name, type) in enumerate(functions_data[function_name].parameters.items()):
        format = f'"{name}": '
        context += format
        result += format

        while True:
            encoded_prompt = llm.encode(context).tolist()[0]
            logits = llm.get_logits_from_input_ids(encoded_prompt)
            picked_token = logits.index(max(logits))
            piece = llm.decode([picked_token])

            context += piece
            result += piece

            if ',' in piece or '}' in piece:
                break

    result = result.rstrip(', }\n') + '}'
    return result

print(get_fn_params("fn_substitute_string_with_regex", functions_data, "Replace all vowels in 'Programming is fun' with asterisks"))