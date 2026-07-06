import json
from typing import Dict, List, Any
import sys
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
)


class ParameterModel(BaseModel):
    """Pydantic model representing a single function parameter.

    Attributes:
        type: The data type of the parameter. Must be one of
            'string', 'number', 'integer', or 'boolean'.
    """
    model_config = {"extra": "forbid"}
    type: str

    @field_validator("type")
    @classmethod
    def valid_type(cls, type: str) -> str:
        """Validate that the parameter type is one of the accepted values."""
        valid_types: List[str] = [
            "string", "number", "integer", "boolean"]
        if type.lower() not in valid_types:
            raise ValueError("Error: Invalid type")
        return type


class ReturnModel(BaseModel):
    """Pydantic model representing the return type of a function.

    Attributes:
        type: The data type of the return value. Must be one of
            'string', 'number', 'integer', or 'boolean'.
    """
    model_config = {"extra": "forbid"}
    type: str

    @field_validator("type")
    @classmethod
    def valid_type(cls, type: str) -> str:
        """Validate that the return type is one of the accepted values."""
        valid_types: List[str] = [
            "string", "number", "integer", "boolean"]
        if type.lower() not in valid_types:
            raise ValueError("Invalid type")
        return type


class FunctionDefinitionModel(BaseModel):
    """Pydantic model representing a complete function definition.

    Attributes:
        name: The unique name of the function. Must not be blank.
        description: A human-readable description of what the function does.
            Must not be blank.
        parameters: A mapping of parameter names to their ParameterModel
            descriptors. Parameter names must not be blank.
        returns: The ReturnModel describing the function's return type.
    """
    model_config = {"extra": "forbid"}
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, ParameterModel]
    returns: ReturnModel

    @field_validator("name", "description")
    @classmethod
    def not_empty(cls, value: str) -> str:
        """Validate that name and description fields are not blank."""
        if not value.strip():
            raise ValueError("Error: filed must not be blank")
        return value

    @field_validator("parameters")
    @classmethod
    def valid_param_name(
        cls,
        value: Dict[str, ParameterModel]
    ) -> Dict[str, ParameterModel]:
        """Validate that no parameter name in the mapping is blank."""
        for name in value:
            if not name.strip():
                raise ValueError("Error: parameter name can't be empty")
        return value


class FunctionCallingTests(BaseModel):
    """Pydantic model representing a single test prompt entry.

    Attributes:
        prompt: The natural-language request to be processed.
        Must not be blank.
    """
    model_config = {"extra": "forbid"}
    prompt: str

    @field_validator("prompt")
    @classmethod
    def valid_prompt(cls, prompt: str) -> str:
        """Validate that the prompt field is not blank."""
        if not prompt.strip():
            raise ValueError("Error: prompt can't be empty")
        return prompt


def load_and_validate_functions(
    filepath: str
) -> List[FunctionDefinitionModel]:
    """Load and validate a function definitions JSON file.

    Reads a JSON array from the given filepath and validates each entry
    against FunctionDefinitionModel. Exits with an error message if the
    file is not found, contains invalid JSON, or fails schema validation.

    Args:
        filepath: Path to the JSON file containing function definitions.

    Returns:
        A list of validated FunctionDefinitionModel instances.
    """
    try:
        with open(filepath, 'r') as f:
            data: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        sys.exit("Error: functions definition file not found")
    except json.JSONDecodeError:
        sys.exit("Error: invalid JSON in functions definition file")

    try:
        validated_functions: List[FunctionDefinitionModel] = []
        for func_dict in data:
            func_obj: FunctionDefinitionModel = FunctionDefinitionModel(
                **func_dict)
            validated_functions.append(func_obj)

        return validated_functions
    except ValidationError as e:
        sys.exit(f"{e}")
    except TypeError:
        sys.exit("Invalid JSON valid data in function definition file")


def load_and_validate_function_calling_tests(
    filepath: str
) -> List[FunctionCallingTests]:
    """Load and validate a function calling tests JSON file.

    Reads a JSON array from the given filepath and validates each entry
    against FunctionCallingTests. Exits with an error message if the file
    is not found, contains invalid JSON, or fails schema validation.

    Args:
        filepath: Path to the JSON file containing test prompts.

    Returns:
        A list of validated FunctionCallingTests instances.
    """

    try:
        with open(filepath, 'r') as f:
            data: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        sys.exit("Error: function calling tests file not found")
    except json.JSONDecodeError:
        sys.exit("Error: invalid JSON in tests file")

    try:
        validate_prompts: List[FunctionCallingTests] = []
        for prompt in data:
            prompt_obj: FunctionCallingTests = FunctionCallingTests(**prompt)
            validate_prompts.append(prompt_obj)

        return validate_prompts
    except ValidationError as e:
        sys.exit(f"{e}")
    except TypeError:
        sys.exit("Invalid JSON valid data in function definition file")
