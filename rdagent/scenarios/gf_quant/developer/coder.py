"""Lightweight coder for the GF remote factor DSL.

This module owns the local operator catalog: it validates a proposed expression
against the same grammar the remote worker enforces, so an invalid candidate is
rejected before any SSH round-trip.  The worker keeps its own copy of the table
because it is shipped to the data host as a standalone file;
``test_operators.py`` asserts the two catalogs never drift apart.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from rdagent.core.developer import Developer
from rdagent.core.exception import FactorEmptyError

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
    "industry",
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
_RAW_FINANCIAL_ARGUMENTS: dict[str, tuple[int, ...]] = {
    "YOY": (0,),
    "QOQ": (0,),
    "YOYOQOQ": (0,),
    "Delta": (0,),
    "SUD": (0,),
    "DeltaProp": (0,),
    "DeltaA2B": (0,),
    "YOYA_YOYB": (0, 1),
    "DeltaRegA2B": (0,),
    "YOYChange": (0,),
    "CAGR": (0,),
}
_INDUSTRY_ARGUMENTS: dict[str, int] = {
    "IndusProp": 1,
    "DeltaProp": 1,
    "cs_industry_rank": 1,
    "cs_industry_demean": 1,
    "cs_industry_zscore": 1,
}


@dataclass(frozen=True)
class ArgRule:
    """Validation for one positional call argument that must be a literal.

    ``minimum``/``maximum`` are exclusive. Windows and shifts use ``kind="int"``;
    fractions such as the winsorize quantile use ``kind="float"``.
    """

    position: int
    kind: str
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class OperatorSpec:
    """Declarative contract of one DSL operator: name, signature and argument rules."""

    name: str
    signature: str
    min_args: int
    max_args: int
    arguments: tuple[ArgRule, ...] = ()


_WINDOW = (ArgRule(1, "int", minimum=1),)
_OPERATOR_LIST: tuple[OperatorSpec, ...] = (
    # Rolling statistics.
    OperatorSpec("ts_mean", "ts_mean(x, n)", 2, 2, _WINDOW),
    OperatorSpec("ts_std", "ts_std(x, n)", 2, 2, _WINDOW),
    OperatorSpec("ts_sum", "ts_sum(x, n)", 2, 2, _WINDOW),
    OperatorSpec("ts_min", "ts_min(x, n)", 2, 2, _WINDOW),
    OperatorSpec("ts_max", "ts_max(x, n)", 2, 2, _WINDOW),
    OperatorSpec("ts_median", "ts_median(x, n)", 2, 2, _WINDOW),
    OperatorSpec("ts_mad", "ts_mad(x, n)", 2, 2, (ArgRule(1, "int", minimum=2),)),
    OperatorSpec("ts_skew", "ts_skew(x, n)", 2, 2, _WINDOW),
    OperatorSpec("ts_kurt", "ts_kurt(x, n)", 2, 2, _WINDOW),
    OperatorSpec("ts_slope", "ts_slope(x, n)", 2, 2, (ArgRule(1, "int", minimum=2),)),
    OperatorSpec("ts_residual", "ts_residual(x, n)", 2, 2, (ArgRule(1, "int", minimum=2),)),
    OperatorSpec("ts_upside_std", "ts_upside_std(x, n)", 2, 2, (ArgRule(1, "int", minimum=2),)),
    OperatorSpec("ts_downside_std", "ts_downside_std(x, n)", 2, 2, (ArgRule(1, "int", minimum=2),)),
    OperatorSpec("ts_above_mean", "ts_above_mean(x, n)", 2, 2, (ArgRule(1, "int", minimum=2),)),
    OperatorSpec(
        "ts_quantile",
        "ts_quantile(x, n, q)",
        3,
        3,
        (ArgRule(1, "int", minimum=1), ArgRule(2, "float", minimum=0.0, maximum=1.0)),
    ),
    # Rolling normalisation and position.
    OperatorSpec("ts_zscore", "ts_zscore(x, n)", 2, 2, _WINDOW),
    OperatorSpec("ts_rank", "ts_rank(x, n)", 2, 2, _WINDOW),
    OperatorSpec("ts_decay_linear", "ts_decay_linear(x, n)", 2, 2, _WINDOW),
    OperatorSpec("ts_ewm", "ts_ewm(x, n)", 2, 2, _WINDOW),
    OperatorSpec("ts_argmax", "ts_argmax(x, n)", 2, 2, _WINDOW),
    OperatorSpec("ts_argmin", "ts_argmin(x, n)", 2, 2, _WINDOW),
    # Lagged change.
    OperatorSpec("delay", "delay(x, n)", 2, 2, _WINDOW),
    OperatorSpec("delta", "delta(x, n)", 2, 2, _WINDOW),
    OperatorSpec("pct_change", "pct_change(x, n)", 2, 2, _WINDOW),
    # Directed point-in-time financial operators.
    OperatorSpec("Self", "Self(x)", 1, 1),
    OperatorSpec("YOY", "YOY(x)", 1, 1),
    OperatorSpec("QOQ", "QOQ(x)", 1, 1),
    OperatorSpec("YOYOQOQ", "YOYOQOQ(x)", 1, 1),
    OperatorSpec("YOYChange", "YOYChange(x, k)", 2, 2, (ArgRule(1, "int", minimum=0),)),
    OperatorSpec("CAGR", "CAGR(x, n)", 2, 2, (ArgRule(1, "int", minimum=0),)),
    OperatorSpec("Delta", "Delta(x)", 1, 1),
    OperatorSpec("SUD", "SUD(x, lookback)", 1, 2, (ArgRule(1, "int", minimum=1),)),
    OperatorSpec("IndusProp", "IndusProp(x, industry)", 2, 2),
    OperatorSpec("DeltaProp", "DeltaProp(x, industry)", 2, 2),
    OperatorSpec("A2B", "A2B(a, b)", 2, 2),
    OperatorSpec("PercentileA2B", "PercentileA2B(a, b)", 2, 2),
    OperatorSpec("RegA2B", "RegA2B(a, b)", 2, 2),
    OperatorSpec("DeltaA2B", "DeltaA2B(a, b)", 2, 2),
    OperatorSpec("YOYA_YOYB", "YOYA_YOYB(a, b)", 2, 2),
    OperatorSpec("DeltaRegA2B", "DeltaRegA2B(a, b)", 2, 2),
    # Two-panel rolling statistics.
    OperatorSpec("ts_corr", "ts_corr(x, y, n)", 3, 3, (ArgRule(2, "int", minimum=1),)),
    OperatorSpec("ts_cov", "ts_cov(x, y, n)", 3, 3, (ArgRule(2, "int", minimum=1),)),
    # Cross-sectional transforms.
    OperatorSpec("cs_rank", "cs_rank(x)", 1, 1),
    OperatorSpec("cs_zscore", "cs_zscore(x)", 1, 1),
    OperatorSpec("cs_demean", "cs_demean(x)", 1, 1),
    OperatorSpec("cs_scale", "cs_scale(x)", 1, 1),
    OperatorSpec("cs_industry_rank", "cs_industry_rank(x, industry)", 2, 2),
    OperatorSpec("cs_industry_demean", "cs_industry_demean(x, industry)", 2, 2),
    OperatorSpec("cs_industry_zscore", "cs_industry_zscore(x, industry)", 2, 2),
    OperatorSpec("cs_zscore_mad", "cs_zscore_mad(x)", 1, 1),
    # Robust transforms.
    OperatorSpec("signed_log", "signed_log(x)", 1, 1),
    OperatorSpec(
        "winsorize",
        "winsorize(x, p)",
        1,
        2,
        (ArgRule(1, "float", minimum=0.0, maximum=0.5),),
    ),
    # Elementwise math.
    OperatorSpec("abs", "abs(x)", 1, 1),
    OperatorSpec("log", "log(x)", 1, 1),
    OperatorSpec("sqrt", "sqrt(x)", 1, 1),
    OperatorSpec("sign", "sign(x)", 1, 1),
    OperatorSpec("inverse", "inverse(x)", 1, 1),
    OperatorSpec(
        "clip",
        "clip(x, lo, hi)",
        3,
        3,
        (ArgRule(1, "float"), ArgRule(2, "float")),
    ),
)

OPERATORS: dict[str, OperatorSpec] = {spec.name: spec for spec in _OPERATOR_LIST}

OPERATOR_GROUPS: dict[str, tuple[str, ...]] = {
    "Rolling statistics": (
        "ts_mean",
        "ts_std",
        "ts_sum",
        "ts_min",
        "ts_max",
        "ts_median",
        "ts_mad",
        "ts_skew",
        "ts_kurt",
        "ts_slope",
        "ts_residual",
        "ts_upside_std",
        "ts_downside_std",
        "ts_above_mean",
        "ts_quantile",
    ),
    "Rolling normalisation and position": (
        "ts_zscore",
        "ts_rank",
        "ts_decay_linear",
        "ts_ewm",
        "ts_argmax",
        "ts_argmin",
    ),
    "Lagged change": ("delay", "delta", "pct_change"),
    "Directed financial": (
        "Self",
        "YOY",
        "QOQ",
        "YOYOQOQ",
        "YOYChange",
        "CAGR",
        "Delta",
        "SUD",
        "IndusProp",
        "DeltaProp",
        "A2B",
        "PercentileA2B",
        "RegA2B",
        "DeltaA2B",
        "YOYA_YOYB",
        "DeltaRegA2B",
    ),
    "Two-panel rolling": ("ts_corr", "ts_cov"),
    "Cross-sectional": (
        "cs_rank",
        "cs_zscore",
        "cs_demean",
        "cs_scale",
        "cs_industry_rank",
        "cs_industry_demean",
        "cs_industry_zscore",
        "cs_zscore_mad",
    ),
    "Robust transforms": ("signed_log", "winsorize"),
    "Elementwise math": ("abs", "log", "sqrt", "sign", "inverse", "clip"),
}

ALLOWED_FUNCTIONS = frozenset(OPERATORS)

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


def describe_operators() -> str:
    """Render the operator catalog as prompt-ready text, one group per line."""
    return "\n".join(
        f"{group}: " + ", ".join(OPERATORS[name].signature for name in names)
        for group, names in OPERATOR_GROUPS.items()
    )


def _check_arguments(spec: OperatorSpec, node: ast.Call) -> None:
    """Check arity and literal argument rules before any data is loaded."""
    if not spec.min_args <= len(node.args) <= spec.max_args:
        raise ValueError(f"{spec.signature} expects between {spec.min_args} and {spec.max_args} arguments")
    for rule in spec.arguments:
        if rule.position >= len(node.args):
            continue
        argument = node.args[rule.position]
        value = _numeric_literal(argument)
        if value is None:
            raise ValueError(f"{spec.signature} argument {rule.position + 1} must be a numeric literal")
        if rule.kind == "int" and value != int(value):
            raise ValueError(f"{spec.signature} argument {rule.position + 1} must be a whole number")
        if rule.minimum is not None and value <= rule.minimum:
            raise ValueError(f"{spec.signature} argument {rule.position + 1} must be greater than {rule.minimum}")
        if rule.maximum is not None and value >= rule.maximum:
            raise ValueError(f"{spec.signature} argument {rule.position + 1} must be less than {rule.maximum}")
    for position in _RAW_FINANCIAL_ARGUMENTS.get(spec.name, ()):
        argument = node.args[position]
        if not isinstance(argument, ast.Name) or not _FINANCIAL_FIELD.fullmatch(argument.id):
            raise ValueError(f"{spec.signature} argument {position + 1} must be a raw point-in-time financial field")
    if spec.name == "clip" and _numeric_literal(node.args[1]) >= _numeric_literal(node.args[2]):
        raise ValueError("clip(x, lo, hi) requires lo < hi")
    industry_position = _INDUSTRY_ARGUMENTS.get(spec.name)
    if industry_position is not None:
        argument = node.args[industry_position]
        if not isinstance(argument, ast.Name) or argument.id != "industry":
            raise ValueError(f"{spec.signature} requires the industry field as argument {industry_position + 1}")


def _numeric_literal(node: ast.AST) -> float | None:
    """Return a signed numeric literal while rejecting names, bools and expressions."""

    if isinstance(node, ast.Constant) and not isinstance(node.value, bool) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and not isinstance(node.operand.value, bool)
        and isinstance(node.operand.value, (int, float))
    ):
        value = float(node.operand.value)
        return -value if isinstance(node.op, ast.USub) else value
    return None


def validate_dsl_expression(expression: str) -> tuple[str, ...]:
    """Validate the same data-only grammar enforced by the remote worker."""

    expression = expression.strip()
    if not expression:
        raise ValueError("Factor formulation must be a non-empty GF DSL expression")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid GF DSL syntax: {exc.msg}") from exc

    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    fields: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsupported GF DSL element: {type(node).__name__}")
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool) or not isinstance(node.value, (int, float))
        ):
            raise ValueError("GF DSL constants must be numeric")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in OPERATORS:
                name = node.func.id if isinstance(node.func, ast.Name) else type(node.func).__name__
                raise ValueError(f"GF DSL calls must use an allow-listed operator, got {name}")
            if node.keywords:
                raise ValueError("GF DSL keyword arguments are not supported")
            _check_arguments(OPERATORS[node.func.id], node)
        if isinstance(node, ast.Name) and node.id not in OPERATORS:
            if node.id not in DIRECT_FIELDS and not _FINANCIAL_FIELD.fullmatch(node.id):
                raise ValueError(f"Unsupported GF data field: {node.id}")
            fields.add(node.id)
        elif isinstance(node, ast.Name) and node.id not in called:
            raise ValueError(f"Operator {node.id} must be called, not referenced")

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
