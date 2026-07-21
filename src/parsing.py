import json
from typing import Dict, List, Any, Tuple
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
        """Validate that the parameter type is one of the accepted values.

        Args:
            type: The parameter type string to validate.

        Returns:
            The validated type string, unchanged.
        """
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
        """Validate that the return type is one of the accepted values.

        Args:
            type: The return type string to validate.

        Returns:
            The validated type string, unchanged.
        """
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
        """Validate that name and description fields are not blank.

        Args:
            value: The field value to validate.

        Returns:
            The validated value, unchanged.
        """
        if not value.strip():
            raise ValueError("Error: filed must not be blank")
        return value

    @field_validator("parameters")
    @classmethod
    def valid_param_name(
        cls,
        value: Dict[str, ParameterModel]
    ) -> Dict[str, ParameterModel]:
        """Validate that no parameter name in the mapping is blank.

        Args:
            value: Mapping of parameter names to their ParameterModel
                descriptors.

        Returns:
            The validated mapping, unchanged.
        """
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
        """Validate that the prompt field is not blank.

        Args:
            prompt: The prompt string to validate.

        Returns:
            The validated prompt, unchanged.
        """
        if not prompt.strip():
            raise ValueError("Error: prompt can't be empty")
        return prompt


def prevent_duplicates(items: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """Build a dict from key-value pairs, rejecting duplicate keys.

    Args:
        items: A list of (key, value) pairs, as passed by the JSON
            decoder for each object it parses.

    Returns:
        A dict mapping each key to its value.
    """
    check = {}
    for key, value in items:
        if key in check:
            raise ValueError("Error: duplicated parameter name")
        check[key] = value
    return check


def validate_functions(
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
            data: List[Dict[str, Any]] = json.load(
                f, object_pairs_hook=prevent_duplicates)
    except FileNotFoundError:
        sys.exit("Error: functions definition file not found")
    except PermissionError:
        sys.exit("Error: permission to read file is denied")
    except IsADirectoryError:
        sys.exit("Error: the path is a directory")
    except UnicodeDecodeError:
        sys.exit("Error: file is not valid text")
    except ValueError as e:
        sys.exit(f"{e}")
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


def validate_tests(
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
            data: List[Dict[str, Any]] = json.load(
                f, object_pairs_hook=prevent_duplicates)
    except FileNotFoundError:
        sys.exit("Error: function calling tests file not found")
    except PermissionError:
        sys.exit("Error: premission to read file is denied")
    except IsADirectoryError:
        sys.exit("Error: the path is a directory")
    except UnicodeDecodeError:
        sys.exit("Error: file is not valid text")
    except ValueError as e:
        sys.exit(f"{e}")
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
        sys.exit("Invalid JSON valid data in tests file")
