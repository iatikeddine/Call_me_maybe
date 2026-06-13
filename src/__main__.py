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


def is_valid_boolean(value: str) -> bool:
    value = value.strip().lower()
    return (
        "true".startswith(value)
        or "false".startswith(value)
        or value in ("true", "false")
    )


def is_valid_integer(value: str) -> bool:
    value = value.strip()

    if value == "":
        return True

    if value == "-":
        return True

    if value.startswith("-"):
        return value[1:].isdigit()

    return value.isdigit()


def is_valid_number(value: str) -> bool:
    value = value.strip()

    if value == "":
        return True

    if value == "-":
        return True

    try:
        float(value)
        return True
    except ValueError:
        pass

    if value.endswith("."):
        try:
            float(value[:-1])
            return True
        except ValueError:
            return False

    return False


def is_valid_string(value: str) -> bool:
    return len(value) <= 128


def validate_value(value: str, param_type: str) -> bool:
    if param_type == "boolean":
        return is_valid_boolean(value)

    if param_type == "integer":
        return is_valid_integer(value)

    if param_type == "number":
        return is_valid_number(value)

    if param_type == "float":
        return is_valid_number(value)

    if param_type == "string":
        return is_valid_string(value)

    return True


def generate_value(
    encoded_prompt: list[int],
    param_type: str,
    max_tokens: int = 20,
) -> str:

    generated = ""

    for _ in range(max_tokens):

        logits = llm.get_logits_from_input_ids(encoded_prompt)

        while True:
            token = logits.index(max(logits))

            piece = llm.decode([token])

            candidate = generated + piece

            if validate_value(candidate, param_type):
                break

            logits[token] = -float("inf")

        encoded_prompt.append(token)
        generated = candidate

        if param_type == "boolean":
            cleaned = generated.strip().lower()
            if cleaned in ("true", "false"):
                break

    return generated.strip()


def get_fn_params(
    all_functions: dict,
    function_name: str,
    user_prompt: str,
) -> dict:

    function = all_functions[function_name]

    result = {}

    for param_name, param_info in function.parameters.items():

        context = (
            f"function: {function_name}\n"
            f"parameter: {param_name}\n"
            f"type: {param_info.type}\n"
            f"user request: {user_prompt}\n"
            f"value:"
        )

        encoded = llm.encode(context).tolist()[0]

        value = generate_value(
            encoded_prompt=encoded,
            param_type=param_info.type,
        )

        if param_info.type == "number":
            try:
                value = float(value)
            except ValueError:
                pass

        result[param_name] = value

    return result


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

    params = get_fn_params(
        all_functions,
        name,
        prompt.prompt_value
    )

    print(prompt.prompt_value)
    print(name)

    print(
        json.dumps(
            params,
            ensure_ascii=False,
            indent=2
        )
    )

    print()