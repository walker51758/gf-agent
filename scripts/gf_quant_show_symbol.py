"""Print one symbol's source from a Python file, with line numbers.

Built for the two oversized strategy-layer files (the remote worker and the parity
test module), where reading the whole file costs more context than the question
deserves and grepping for a name only yields the definition site.

The tool resolves module-level ``def`` / ``async def`` / ``class`` / assigned
``NAME = ...`` targets with :mod:`ast`, then prints only that span::

    python scripts/gf_quant_show_symbol.py scripts/gf_quant_strategy_worker.py _apply_weight_floor
    python scripts/gf_quant_show_symbol.py test/scenarios/gf_quant/test_strategy_worker_parity.py \\
        test_minimum_holding_period_agrees_end_to_end --context 5

``--context N`` adds N lines above and below the symbol.  A dotted ``Class.method``
target selects a method of that class.  When the symbol is absent the tool lists
close matches if any exist and exits non-zero, so it composes with shell pipelines
without being mistaken for success.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import sys
from pathlib import Path


class SymbolNotFound(LookupError):
    """Raised with the candidate list for a missing symbol."""


def _assigned_names(node: ast.stmt) -> list[str]:
    """Module-level names bound by an assignment / ann-assignment statement."""

    names: list[str] = []
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for element in target.elts:
                    if isinstance(element, ast.Name):
                        names.append(element.id)
    return names


class _ModuleIndex:
    """Every addressable top-level symbol of one parsed module."""

    def __init__(self, tree: ast.Module) -> None:
        self.functions: dict[str, ast.AST] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        self.assignments: dict[str, ast.stmt] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions[node.name] = node
            elif isinstance(node, ast.ClassDef):
                self.classes[node.name] = node
            else:
                for name in _assigned_names(node):
                    self.assignments[name] = node

    def find(self, dotted: str) -> ast.AST:
        head, _, tail = dotted.partition(".")
        if head in self.classes:
            if not tail:
                return self.classes[head]
            for node in self.classes[head].body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == tail:
                    return node
            raise SymbolNotFound(f"{tail!r} is not defined in class {head!r}")
        if tail:
            raise SymbolNotFound(f"{head!r} is not a class in this module, so {dotted!r} cannot resolve")
        for table in (self.functions, self.assignments):
            if head in table:
                return table[head]
        raise SymbolNotFound(f"no top-level function, class or assignment named {head!r}")

    def names(self) -> list[str]:
        dotted = list(self.functions) + list(self.classes) + list(self.assignments)
        dotted += [f"{cls}.{fn}" for cls in self.classes for fn in self._methods_of(cls)]
        return dotted

    def _methods_of(self, cls_name: str) -> list[str]:
        return [
            node.name
            for node in self.classes[cls_name].body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]


def read_module(path: Path) -> tuple[ast.Module, list[str]]:
    """Parse ``path``.  Returns the tree and the source split into lines."""

    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise SystemExit(f"{path}: not valid Python: {exc}") from exc
    return tree, source.splitlines()


def select_span(node: ast.AST, line_count: int, context: int) -> tuple[int, int]:
    """Return the 1-indexed inclusive ``(first, last)`` lines to print."""

    decorators = getattr(node, "decorator_list", [])
    first = min([getattr(node, "lineno", 1)] + [d.lineno for d in decorators])
    last = getattr(node, "end_lineno", getattr(node, "lineno", 1))
    return max(1, first - context), min(line_count, last + context)


def render(path: Path, dotted: str, context: int) -> tuple[str, tuple[int, int]]:
    """Return the printable block and the span it covers."""

    tree, lines = read_module(path)
    index = _ModuleIndex(tree)
    try:
        node = index.find(dotted)
    except SymbolNotFound as exc:
        available = index.names()
        close = difflib.get_close_matches(dotted, available, n=8, cutoff=0.5)
        if not close and available:
            close = sorted(available)[:8]
        hint = ", ".join(close) if close else "none"
        raise SymbolNotFound(f"{path}: {exc}. Close matches: {hint}") from None
    first, last = select_span(node, len(lines), context)
    width = len(str(last))
    body = "\n".join(f"{number:>{width}}\t{lines[number - 1]}" for number in range(first, last + 1))
    return f"{path}:{dotted}  lines {first}-{last} ({last - first + 1} lines)\n{body}", (first, last)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gf_quant_show_symbol",
        description="Print one function/class/assignment from a Python file, with line numbers.",
    )
    parser.add_argument("file", help="path to the Python source file")
    parser.add_argument("symbol", help="top-level name, or Class.method")
    parser.add_argument("--context", type=int, default=0, help="extra lines to print before and after (default 0)")
    args = parser.parse_args(argv)

    if args.context < 0:
        print("--context must not be negative", file=sys.stderr)
        return 2

    path = Path(args.file)
    if not path.is_file():
        print(f"no such file: {path}", file=sys.stderr)
        return 2

    try:
        text, _span = render(path, args.symbol, args.context)
    except SymbolNotFound as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
