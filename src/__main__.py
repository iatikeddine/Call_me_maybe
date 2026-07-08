from llm_sdk import Small_LLM_Model
import argparse
import json
import os
import sys
from .parsing import (
    validate_functions,
    validate_tests,
    FunctionCallingTests,
    FunctionDefinitionModel,
)
from typing import List, Dict, Set, Any, Union


def function_name_context() -> str:
    """
    Build a context string listing all available function names
    and descriptions.

    Returns:
        A formatted string enumerating every function name and description.
    """
    context: str = "Available functions\n"
    for function in functions_list:
        context += function.name + ':' + function.description + '\n'
    return context


def valid_names_tokens(valid_names: List[List[int]]) -> Set[int]:
    """Collect all individual token IDs from a list of encoded function names.

    Args:
        valid_names: A list of token-ID sequences, one per candidate
            function name.

    Returns:
        A set containing every token ID that appears in any of the
        candidate name sequences.
    """
    valid_tokens: Set[int] = set()

    for encoded_name in valid_names:
        for token in encoded_name:
            valid_tokens.add(token)

    return valid_tokens


def get_function_name(prompt: str, valid_names: List[List[int]]) -> List[int]:
    """Use constrained decoding to select the best-matching function name.

    Args:
        prompt: The natural-language user request.
        valid_names: A list of token-ID sequences representing all
            candidate function names.

    Returns:
        The token-ID sequence of the selected function name.
    """
    prompt_context: str = function_name_context()
    prompt_helper: str = "predict function name for this question: "
    prompt_shema: str = '{"function name": "'
    encoded_prompt: List[int] = llm.encode(
        prompt_context + prompt_helper + prompt + prompt_shema).tolist()[0]

    token_position: int = 0
    valid_names_copy: List[List[int]] = valid_names.copy()

    while (len(valid_names_copy) > 1):
        logits: List[float] = llm.get_logits_from_input_ids(encoded_prompt)
        valid_tokens: Set[int] = valid_names_tokens(valid_names_copy)

        vocab_len: int = len(logits)
        for x in range(vocab_len):
            if x not in valid_tokens:
                logits[x] = -float('inf')

        picked_token: int = logits.index(max(logits))

        valid_names_copy = [
            name for name in valid_names_copy
            if len(name) > token_position
            if picked_token == name[token_position]
        ]

        encoded_prompt.append(picked_token)
        token_position += 1

    return valid_names_copy[0]


def valid_string(encoded_prompt: List[int]) -> str:
    """Generate a constrained JSON string value using the LLM.

    Args:
        encoded_prompt: The current list of input token IDs representing
            the generation context built so far.

    Returns:
        A string of the form ``"<content>"`` including the surrounding
        double-quote characters.
    """
    quote_id: int = vocab_content['"']
    result: str = ''
    max_tokens = 100

    logits: List[float] = llm.get_logits_from_input_ids(encoded_prompt)
    vocab_len: int = len(logits)
    for x in range(vocab_len):
        if x != quote_id:
            logits[x] = -float('inf')

    picked_token: int = logits.index(max(logits))
    encoded_prompt.append(picked_token)
    result += llm.decode([picked_token])

    generated_tokens: int = 0
    while generated_tokens < max_tokens:
        logits = llm.get_logits_from_input_ids(encoded_prompt)
        picked_token = logits.index(max(logits))
        decoded: str = llm.decode([picked_token])

        if '"' in decoded:
            quote_position: int = decoded.index('"')
            valid: str = decoded[:quote_position + 1]
            result += valid

            if valid == '"':
                encoded_prompt.append(quote_id)
            else:
                valid_ids: List[int] = llm.encode(valid).tolist()[0]
                encoded_prompt.extend(valid_ids)
            return result

        encoded_prompt.append(picked_token)
        result += llm.decode([picked_token])
        generated_tokens += 1

    result += '"'
    encoded_prompt.append(quote_id)
    return result


def valid_number(encoded_prompt: List[int]) -> str:
    """Generate a constrained numeric value using the LLM.

    Args:
        encoded_prompt: The current list of input token IDs representing
            the generation context built so far.

    Returns:
        A string containing only the raw numeric characters, without
        surrounding quotes or trailing delimiters.
    """
    valid_tokens: List[str] = [
        '0', '1', '2', '3', '4', '5', '6',
        '7', '8', '9', '-', '.', ',', '}'
    ]
    valid_tokens_encoded: List[int] = [
        llm.encode(token).tolist()[0][0]
        for token in valid_tokens]
    result: str = ""
    max_tokens = 100

    generated_tokens: int = 0
    while generated_tokens < max_tokens:
        logits: List[float] = llm.get_logits_from_input_ids(encoded_prompt)
        for x in range(len(logits)):
            if x not in valid_tokens_encoded:
                logits[x] = -float('inf')

        picked_token: int = logits.index(max(logits))
        decoded: str = llm.decode([picked_token])
        if ',' in decoded or '}' in decoded:
            return result

        encoded_prompt.append(picked_token)
        result += llm.decode([picked_token])
        generated_tokens += 1

    return result


def valid_bool(encoded_prompt: List[int]) -> str:
    """Generate a constrained boolean value using the LLM.

    Args:
        encoded_prompt: The current list of input token IDs representing
            the generation context built so far.

    Returns:
        The string '0' or '1' representing the predicted boolean value.
    """
    valid_bool: List[str] = ['0', '1']
    valid_tokens: List[int] = [
        llm.encode(x).tolist()[0][0]
        for x in valid_bool]

    logits: List[float] = llm.get_logits_from_input_ids(encoded_prompt)

    vocab_len: int = len(logits)
    for x in range(vocab_len):
        if x not in valid_tokens:
            logits[x] = -float('inf')

    picked_token: int = logits.index(max(logits))
    encoded_prompt.append(picked_token)
    result: str = llm.decode([picked_token])
    return result


def get_fn_params(
    function_name: str,
    functions_data: Dict[str, Any],
    prompt: str
) -> str:
    """Generate a JSON object string containing all parameters for a function.

    Args:
        function_name: The name of the function whose parameters are to be
            generated.
        functions_data: A mapping of function names to their
            FunctionDefinitionModel instances.
        prompt: The natural-language user request, used to guide the LLM
            in predicting the correct argument values.

    Returns:
        A JSON object string of the form ``{"param": value, ...}``
        containing all required parameters with their predicted values.
    """
    params_dict: Dict[str, str] = {
        key: value.type
        for key, value in functions_data[function_name].parameters.items()}

    params_len: int = len(params_dict)

    context: str = (
        f"function name: {function_name}\n"
        f"function purpose: {functions_data[function_name].description}\n"
        f"number of parameters: {params_len}\n"
        f"question: {prompt}\n"
        f"predict the correct function parameters in JSON format ")

    context += '{'
    result: str = '{'

    param_position: int = 0
    for name, param_type in params_dict.items():
        field: str = f'"{name}": '
        context += field
        result += field

        encoded_prompt: List[int] = llm.encode(context).tolist()[0]
        if param_type.lower() == "string":
            value: str = valid_string(encoded_prompt)

        elif param_type.lower() in ["number", "integer"]:
            value = valid_number(encoded_prompt)

        elif param_type.lower() == "boolean":
            value = valid_bool(encoded_prompt)

        result += value
        context += value

        if param_position == params_len - 1:
            result += '}'
        else:
            result += ','
            context += ','
        param_position += 1

    return result


def cast_value(value: str, type: str) -> Union[str, int, float, bool]:
    """Cast a string value to its declared Python type.

    Args:
        value: The raw string representation of the parameter value.
        type: The declared type from the function definition. One of
            'number', 'integer', 'boolean', or 'string'.

    Returns:
        The value converted to ``float``, ``int``, ``bool``, or ``str``
        depending on ``type``.
    """
    if type == "number":
        return float(value)
    elif type == "integer":
        return int(value)
    elif type == "boolean":
        return bool(int(value))
    return value


def clean_text(text: str) -> str:
    """Cleans a string by double-escaping invalid JSON escape sequences.

    Args:
        text (str): The raw input string containing potentially
        invalid escapes.
    Returns:
        str: The sanitized string with invalid escapes safely double-escaped.
    """
    valid_escapes: Set[str] = set(
        ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']
    )

    final_result: List[str] = []

    i: int = 0
    text_len: int = len(text)
    while i < text_len:
        current: str = text[i]

        if current == '\\':
            if i + 1 == text_len:
                final_result.append('\\\\')
                break

            next_char = text[i + 1]
            if next_char in valid_escapes:
                final_result.append('\\')
                final_result.append(next_char)
            else:
                final_result.append('\\\\')
                final_result.append(next_char)

            i += 2
        else:
            final_result.append(current)
            i += 1

    return "".join(final_result)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Function calling with constrained decoding"
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the function calling tests JSON file",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path for the output JSON file",
    )
    args = parser.parse_args()
    try:
        if not args.output.endswith(".json"):
            raise ValueError("invalid ouput path does not contain .json")
    except ValueError as e:
        sys.exit(f"{e}")

    functions_list: List[FunctionDefinitionModel] = validate_functions(
        args.functions_definition
    )
    prompts: List[FunctionCallingTests] = validate_tests(args.input)

    try:
        llm: Small_LLM_Model = Small_LLM_Model()
    except Exception as e:
        sys.exit(f"Error: failed to load the LLM model: {e}")

    try:
        vocab_file: str = llm.get_path_to_vocab_file()
        with open(vocab_file, 'r') as f:
            vocab_content: Dict[str, int] = json.load(f)
    except FileNotFoundError:
        sys.exit("Error: vocabulary file not found")
    except json.JSONDecodeError as e:
        sys.exit(f"Error: invalid JSON in vocabulary file {e}")

    try:
        valid_names: List[List[int]] = [
            llm.encode(function.name).tolist()[0]
            for function in functions_list
        ]
    except Exception as e:
        sys.exit(f"Error: failed to encode function names: {e}")

    predicted_function_names: List[str] = []
    try:
        for prompt in prompts:
            result: List[int] = get_function_name(prompt.prompt, valid_names)
            predicted_function_names.append(clean_text(llm.decode(result)))
    except Exception as e:
        sys.exit(f"{e}")

    functions_data: Dict[str, FunctionDefinitionModel] = {
        function.name: function for function in functions_list}

    predicted_functions_params: List[str] = []
    try:
        for name, prompt in zip(predicted_function_names, prompts):
            predicted_functions_params.append(
                clean_text(get_fn_params(name, functions_data, prompt.prompt)))
    except Exception as e:
        sys.exit(f"{e}")

    results: List[Dict[str, Any]] = []
    try:
        for name, params, prompt in zip(
            predicted_function_names, predicted_functions_params, prompts
        ):
            params_to_dict: Dict[str, Any] = json.loads(params)
            if len(params_to_dict) == 1:
                (param_name, param_value), = params_to_dict.items()
                param_type = functions_data[name].parameters[param_name]
                params_to_dict[param_name] = cast_value(
                    param_value, param_type.type)
            else:
                for param_name, param_value in params_to_dict.items():
                    param_type = functions_data[name].parameters[param_name]
                    params_to_dict[param_name] = cast_value(
                        param_value, param_type.type)

            results.append({
                "prompt": prompt.prompt,
                "name": name,
                "parameters": params_to_dict,
            })
    except (ValueError, KeyError):
        sys.exit("Error: type casting failed")
    except json.JSONDecodeError:
        sys.exit("Error: invalid JSON format")

    output_path: str = args.output
    try:
        output_dir: str = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    except (OSError, IOError) as e:
        sys.exit(f"Error: could not write output file '{output_path}': {e}")

    print(f"Done. Results written to '{output_path}'")
