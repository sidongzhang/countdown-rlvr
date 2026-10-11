"""Rule-based rewards for the Countdown arithmetic task.

The original TinyZero scorer evaluated model output with ``eval``. This
version parses the expression with :mod:`ast` and evaluates a deliberately
small arithmetic grammar, which makes the verifier deterministic and avoids
executing generated Python code.
"""

import ast
import re
from collections import Counter
from fractions import Fraction
from typing import Any, Dict, Iterable, Optional, Tuple


ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def _assistant_text(solution_str: str) -> str:
    """Drop the prompt so examples in the prompt cannot be scored as answers."""
    if "<|im_start|>assistant" in solution_str:
        return solution_str.rsplit("<|im_start|>assistant", 1)[-1]
    if "Assistant:" in solution_str:
        return solution_str.rsplit("Assistant:", 1)[-1]
    return solution_str


def extract_solution(solution_str: str) -> Optional[str]:
    """Return the last answer-tag payload emitted by the assistant."""
    matches = ANSWER_PATTERN.findall(_assistant_text(solution_str))
    if not matches:
        return None
    answer = matches[-1].strip()
    return answer or None


def _evaluate_node(node: ast.AST, leaves: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, leaves)

    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        leaves.append(node.value)
        return Fraction(node.value)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_node(node.operand, leaves)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _evaluate_node(node.left, leaves)
        right = _evaluate_node(node.right, leaves)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ZeroDivisionError("division by zero")
        return left / right

    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def evaluate_equation(equation_str: str) -> Tuple[Optional[Fraction], list[int]]:
    """Safely evaluate an expression and return its integer leaves."""
    try:
        tree = ast.parse(equation_str, mode="eval")
        leaves: list[int] = []
        value = _evaluate_node(tree, leaves)
        return value, leaves
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None, []


def validate_equation(equation_str: str, available_numbers: Iterable[int]) -> bool:
    """Check that every supplied number is used exactly once."""
    value, leaves = evaluate_equation(equation_str)
    return value is not None and Counter(leaves) == Counter(int(number) for number in available_numbers)


def compute_score(
    solution_str: str,
    ground_truth: Dict[str, Any],
    method: str = "strict",
    format_score: float = 0.1,
    validity_score: float = 0.2,
    score: float = 1.0,
    return_details: bool = False,
):
    """Compute a layered, verifiable reward.

    Reward levels are intentionally ordered as follows:

    * no answer tags: 0
    * answer tags but an invalid expression: ``format_score``
    * valid expression using every number exactly once: ``validity_score``
    * valid expression equal to the target: ``score``

    ``method`` is kept for compatibility with the original TinyZero API.
    When ``return_details`` is true, the reward manager also receives binary
    diagnostics for WandB logging.
    """
    del method
    target = Fraction(int(ground_truth["target"]))
    numbers = [int(number) for number in ground_truth["numbers"]]
    equation = extract_solution(solution_str)

    details = {
        "score": 0.0,
        "format_ok": 0.0,
        "syntax_ok": 0.0,
        "numbers_ok": 0.0,
        "valid_ok": 0.0,
        "correct": 0.0,
    }

    if equation is not None:
        details["format_ok"] = 1.0
        details["score"] = float(format_score)
        value, leaves = evaluate_equation(equation)
        if value is not None:
            details["syntax_ok"] = 1.0
            if Counter(leaves) == Counter(numbers):
                details["numbers_ok"] = 1.0
                details["valid_ok"] = 1.0
                details["score"] = float(validity_score)
                if value == target:
                    details["correct"] = 1.0
                    details["score"] = float(score)

    return details if return_details else details["score"]
