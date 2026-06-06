import json
from typing import Dict, List
from pydantic import BaseModel, Field

class ParameterModel(BaseModel):
    type: str

class ReturnModel(BaseModel):
    type: str

class FunctionDefinitionModel(BaseModel):
    name: str = Field(min_length=3, max_length=50)
    description: str
    parameters: Dict[str, ParameterModel]
    returns: ReturnModel

class FunctionCallingTests(BaseModel):
    prompt: str
    prompt_value: str

def load_and_validate_functions(filepath: str) -> List[FunctionDefinitionModel]:
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        validated_functions = []
        for func_dict in data:
            func_obj = FunctionDefinitionModel(**func_dict)
            validated_functions.append(func_obj)

        return validated_functions

    except Exception as e:
        print(f"Error: {e}")
    return []

def load_and_validate_function_calling_tests(filepath: str) -> List[FunctionCallingTests]:
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        validate_prompts = []
        for prompt in data:
            for key, value in prompt.items():
                prompt_obj = FunctionCallingTests(prompt=key, prompt_value=value)
                validate_prompts.append(prompt_obj)
        return validate_prompts
    except Exception as e:
        print(f"Error: {e}")
        return []

functions_list = load_and_validate_functions('data/input/functions_definition.json')
prompts = load_and_validate_function_calling_tests('data/input/function_calling_tests.json')
