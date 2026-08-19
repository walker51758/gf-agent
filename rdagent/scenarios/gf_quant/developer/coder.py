"""Lightweight coder for the GF remote factor DSL."""

from __future__ import annotations

import ast
import re

from rdagent.core.developer import Developer
from rdagent.core.exception import FactorEmptyError


ALLOWED_FUNCTIONS = {
    "cs_rank",
    "delay",
    "delta",
    "pct_change",
    "signed_log",
    "ts_mean",
    "ts_std",
    "winsorize",
}

DIRECT_FIELDS = {
    "adjclose",
    "adjhigh",
    "adjlow",
    "adjopen",
    "amount",
    "close",
    "floatshare",
    "fmv",
    "high",
    "low",
    "mv",
    "open",
    "pb",
    "pe",
    "ps",
    "totalshare",
    "turnover",
    "volume",
    "yclose",
}

_FINANCIAL_FIELD = re.compile(r"^(balance|cashflow|income)__[A-Za-z0-9_]+$")
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.UAdd,
    ast.USub,
)


def validate_dsl_expression(expression: str) -> tuple[str, ...]:
    """Validate the same data-only grammar enforced by the remote worker."""

    expression = expression.strip()
    if not expression:
        raise ValueError("Factor formulation must be a non-empty GF DSL expression")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid GF DSL syntax: {exc.msg}") from exc

    fields: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsupported GF DSL element: {type(node).__name__}")
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool) or not isinstance(node.value, (int, float))
        ):
            raise ValueError("GF DSL constants must be numeric")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
                raise ValueError("GF DSL calls must use an allow-listed function")
            if node.keywords:
                raise ValueError("GF DSL keyword arguments are not supported")
        if isinstance(node, ast.Name) and node.id not in ALLOWED_FUNCTIONS:
            if node.id not in DIRECT_FIELDS and not _FINANCIAL_FIELD.fullmatch(node.id):
                raise ValueError(f"Unsupported GF data field: {node.id}")
            fields.add(node.id)

    if not fields:
        raise ValueError("GF DSL expression must reference at least one data field")
    return tuple(sorted(fields))


class GFFactorCoder(Developer):
    """Treat the proposed formulation as executable DSL; never generate local Python."""

    def develop(self, exp):  # type: ignore[no-untyped-def]
        if not exp.sub_tasks:
            raise FactorEmptyError("The proposal did not contain any GF factor tasks")
        for task in exp.sub_tasks:
            try:
                fields = validate_dsl_expression(str(task.factor_formulation))
            except ValueError as exc:
                raise FactorEmptyError(f"Invalid GF DSL for {task.factor_name}: {exc}") from exc
            task.factor_formulation = str(task.factor_formulation).strip()
            task.factor_implementation = True
            task.dsl_fields = fields
        return exp
