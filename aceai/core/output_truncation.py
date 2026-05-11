DEFAULT_TRUNCATED_OUTPUT_TOKEN_BUDGET = 2500
CHARS_PER_TOKEN = 4


def truncate_output(
    text: str,
    *,
    token_budget: int = DEFAULT_TRUNCATED_OUTPUT_TOKEN_BUDGET,
) -> str:
    if type(text) is not str:
        raise TypeError("truncated output must be str")
    if type(token_budget) is not int:
        raise TypeError("token_budget must be int")
    if token_budget < 1:
        raise ValueError("token_budget must be positive")

    char_budget = token_budget * CHARS_PER_TOKEN
    if len(text) <= char_budget:
        return text

    marker = _truncation_marker(text[char_budget:])
    retained_budget = max(1, char_budget - len(marker))
    head_budget = max(1, retained_budget // 2)
    tail_budget = max(1, retained_budget - head_budget)
    return text[:head_budget] + marker + text[-tail_budget:]


def _truncation_marker(omitted: str) -> str:
    omitted_tokens = max(1, len(omitted) // CHARS_PER_TOKEN)
    return f"\n... {omitted_tokens} tokens truncated ...\n"
