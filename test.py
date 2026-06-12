        generated = generated.strip().strip('"')
        if not validate_param(generated, param_info.type):
            raise ValueError(
                f"Invalid value '{generated}' "
                f"for parameter '{param_name}' "
                f"(expected {param_info.type})"
            )