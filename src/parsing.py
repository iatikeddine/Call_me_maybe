import json
from typing import Dict, List, Any
import sys
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)


class ParameterModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str

    @field_validator("type")
    @classmethod
    def valid_type(cls, type: str) -> str:
        valid_types: List[str] = [
            "string", "number", "integer", "boolean", "bool"]
        if type.lower() not in valid_types:
            raise ValueError("Invalid type")
        return type


class ReturnModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str

    @field_validator("type")
    @classmethod
    def valid_type(cls, type: str) -> str:
        valid_types: List[str] = [
            "string", "number", "integer", "boolean", "bool"]
        if type.lower() not in valid_types:
            raise ValueError("Invalid type")
        return type


class FunctionDefinitionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, ParameterModel]
    returns: ReturnModel

    @field_validator("name", "description")
    @classmethod
    def not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("filed must not be blank")
        return value

    @field_validator("parameters")
    @classmethod
    def valid_param_name(
        cls,
        value: Dict[str, ParameterModel]
    ) -> Dict[str, ParameterModel]:
        for name in value:
            if not name.strip():
                raise ValueError("parameter name can't be empty")
        return value


class FunctionCallingTests(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str

    @field_validator("prompt")
    @classmethod
    def valid_prompt(cls, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt can't be empty")
        return prompt


def load_and_validate_functions(
    filepath: str
) -> List[FunctionDefinitionModel]:
    try:
        with open(filepath, 'r') as f:
            data: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Error: functions definition file not found: '{filepath}'")
    except json.JSONDecodeError as e:
        sys.exit(
            f"Error: invalid JSON in functions definition file '{filepath}': {e}")

    try:
        validated_functions: List[FunctionDefinitionModel] = []
        for func_dict in data:
            func_obj: FunctionDefinitionModel = FunctionDefinitionModel(
                **func_dict
            )
            validated_functions.append(func_obj)
        return validated_functions
    except ValidationError as e:
        sys.exit(
            f"Error: functions definition schema validation failed "
            f"in '{filepath}': {e}")


def load_and_validate_function_calling_tests(
    filepath: str
) -> List[FunctionCallingTests]:
    try:
        with open(filepath, 'r') as f:
            data: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Error: function calling tests file not found: '{filepath}'")
    except json.JSONDecodeError as e:
        sys.exit(
            f"Error: invalid JSON in tests file '{filepath}': {e}")

    try:
        validate_prompts: List[FunctionCallingTests] = []
        for prompt in data:
            prompt_obj: FunctionCallingTests = FunctionCallingTests(**prompt)
            validate_prompts.append(prompt_obj)
        return validate_prompts
    except ValidationError as e:
        sys.exit(
            f"Error: test prompts schema validation failed "
            f"in '{filepath}': {e}")

