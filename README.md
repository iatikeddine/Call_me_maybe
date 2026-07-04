*This project has been created as part of the 42 curriculum by iatikeddine.*

# Call me maybe

## Description

**Call me maybe** is a function calling tool that translates natural language prompts into structured JSON function calls using a small language model (Qwen3-0.6B).

Given a prompt like `"What is the sum of 2 and 3?"`, the program does not answer the question. Instead, it identifies the right function to call and its arguments:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": { "a": 2.0, "b": 3.0 }
}
```

The core challenge is guaranteeing **100% valid JSON output** from a 0.6B parameter model. This is achieved through **constrained decoding**: at every generation step the model's token probabilities are filtered so only tokens that maintain valid structure are allowed.

---

## Instructions

### Requirements

- Python 3.12+
- [uv] package manager

### Installation

```bash
uv sync
```

This installs all dependencies declared in `pyproject.toml` into a local virtual environment.

### Running the program

```bash
uv run python -m src
```

By default the program reads from `data/input/` and writes to `data/output/function_calling_results.json`.

Custom paths can be provided via arguments:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Using the Makefile

```bash
make install  # install dependencies (uv sync)
make run      # run with default paths
make clean    # remove __pycache__ and .mypy_cache
make fclean   # clean + remove .venv and output file
make re       # fclean + all
```

---

## Example Usage

```bash
# Run with default paths
uv run python -m src

# Run with custom input files
uv run python -m src \
  --functions_definition path/to/functions.json \
  --input path/to/tests.json \
  --output path/to/results.json
```

Example output written to `data/output/function_calling_results.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": { "s": "hello" }
  }
]
```

---

## Algorithm Explanation

The program uses **constrained decoding** at two stages.

### Stage 1 — Function name selection

All candidate function names are encoded into token ID sequences. At each generation step:

1. The model produces logits (probability scores) over the entire vocabulary.
2. Only tokens that are valid continuations of at least one remaining candidate name are kept; all others are set to `-inf`.
3. The highest-scoring valid token is picked.
4. Candidates that do not match the picked token at the current position are eliminated.
5. This repeats until exactly one candidate remains.

This guarantees the output is always a real function name from the definitions file — the model cannot hallucinate a name.

### Stage 2 — Parameter value generation

For each parameter, the declared type in `functions_definition.json` determines which constrained decoder is used:

- **string** (`valid_string`): Forces the first token to be `"`, then generates freely until a closing `"` is detected or after a safe boundry. The closing quote is always guaranteed.
- **number / integer** (`valid_number`): Restricts every token to the set `{0-9, -, .}`. Stops when `,` or `}` appears.
- **boolean** (`valid_bool`): Restricts selection to exactly two tokens: `0` (false) and `1` (true). Picks the higher-scoring one.

After all parameters are generated, backslashes in the result are escaped and the final JSON string is parsed and cast to the correct Python types.

---

## Design Decisions

- **Pydantic for input validation**: Both input files (`functions_definition.json` and `function_calling_tests.json`) are validated with Pydantic models on load. Malformed input exits early with a clear error message rather than crashing mid-run.
- **Greedy decoding (argmax)**: The highest-probability valid token is always picked. This is deterministic and avoids sampling randomness, which is preferable when the goal is correctness over diversity.
- **Context re-encoding per parameter**: For multi-parameter functions, the generation context is rebuilt and re-encoded from scratch for each parameter. This keeps the context accurate as values accumulate.
- **Type casting after generation**: String values from the LLM are cast to their declared Python types (`float`, `int`, `bool`) as a final step, separate from generation.
- **argparse for flexibility**: Input and output paths are configurable via CLI flags with sensible defaults, matching the usage pattern specified in the subject.

---

## Performance Analysis

- **Accuracy**: Achieves approximately 90%+ correct function name selection and argument extraction on the provided test set.
- **JSON validity**: 100% of outputs are parseable JSON. Constrained decoding structurally prevents malformed output.
- **Speed**: Processes all 11 test prompts in under 5 minutes on standard hardware. The main bottleneck is the LLM inference calls, one per token per parameter.
- **Reliability**: The program never crashes on valid input. All file I/O, model loading, and type casting are wrapped in specific error handlers that print informative messages and exit cleanly.

---

## Challenges Faced

- **Token boundary mismatches**: A function name like `fn_add_numbers` may be split into multiple tokens differently depending on context. The solution was to encode each candidate name directly and work entirely in token ID space, never assuming a name maps to a single token.
- **Multi-token delimiters**: Characters like `,` and `}` can appear as part of a larger token (e.g. `, ` or `},`). The number decoder checks for their presence in the decoded string rather than checking for an exact token match.
- **String termination**: A closing `"` can appear mid-token (e.g. a token decoding to `hello"`). The string decoder uses `str.index('"')` to find the first quote in the decoded text and slices up to and including it.
- **Backslash handling**: Generated strings may contain backslash characters. A post-processing `replace("\\", "\\\\")` step on the full result ensures they are valid JSON escape sequences.

---

## Testing Strategy

- **Manual testing**: The provided `function_calling_tests.json` was used throughout development to check name selection and parameter extraction.
- **Edge case prompts**: Additional prompts were written to stress-test the string decoder — inputs containing quotes, backslashes, long values, and special characters — to verify JSON format remained intact.

---

## Resources

### References

- [Qwen3 model card — Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
- [JSON specification (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259)
- [Outlines library — constrained generation concepts](https://github.com/outlines-dev/outlines)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)
- [Byte Pair Encoding tokenization — Hugging Face NLP course](https://huggingface.co/learn/nlp-course/chapter6/5)

### AI Usage

AI was used for two purposes in this project:

1. **Learning**: Used to understand how LLMs work internally — specifically how tokenization, logits, and constrained decoding interact. This helped build intuition before writing the generation logic.
2. **Docstrings**: Used to write PEP 257 compliant docstrings for all functions and classes in `src/__main__.py` and `src/parsing.py`.
