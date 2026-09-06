"""Data-only Deforum keyframes; expressions use a bounded arithmetic interpreter."""

from __future__ import annotations

import ast
from bisect import bisect_right
import math
import operator
import re
from typing import Any


FUNCTIONS = {name: getattr(math, name) for name in ("sin", "cos", "tan", "sqrt", "exp", "log", "floor", "ceil")}
FUNCTIONS.update(abs=abs, min=min, max=max)
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow}


class Expression:
    def __init__(self, text: str):
        if len(text) > 1024:
            raise ValueError("Schedule expressions are limited to 1024 characters.")
        try:
            self.tree = ast.parse(text.strip(), mode="eval").body
        except (SyntaxError, RecursionError) as error:
            raise ValueError("Invalid schedule expression.") from error
        nodes = list(ast.walk(self.tree))
        if len(nodes) > 128:
            raise ValueError("Schedule expressions are limited to 128 syntax nodes.")
        self.dynamic = any(isinstance(node, ast.Name) and node.id == "t" for node in nodes)

    def at(self, t: int, max_f: int, fps: float) -> float:
        variables = {"t": t, "max_f": max_f, "fps": fps, "pi": math.pi, "e": math.e}

        def evaluate(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                value = float(node.value)
            elif isinstance(node, ast.Name) and node.id in variables:
                value = variables[node.id]
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 32:
                    raise ValueError("Schedule exponents must be in [-32,32].")
                value = OPERATORS[type(node.op)](left, right)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id in FUNCTIONS and not node.keywords and 1 <= len(node.args) <= 4):
                value = FUNCTIONS[node.func.id](*(evaluate(arg) for arg in node.args))
            else:
                raise ValueError("Schedules accept only numeric arithmetic and documented math functions.")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e12:
                raise ValueError("Schedule values must be finite and within +/-1e12.")
            return value

        try:
            return float(evaluate(self.tree))
        except (ArithmeticError, TypeError, RecursionError) as error:
            raise ValueError("Invalid arithmetic in schedule expression.") from error


def _entries(text: str) -> list[str]:
    depth, start, entries = 0, 0, []
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            entries.append(text[start:index].strip())
            start = index + 1
        if depth < 0:
            raise ValueError("Unbalanced schedule parentheses.")
    entries.append(text[start:].strip())
    if depth or any(not entry for entry in entries):
        raise ValueError("Empty keyframe or unbalanced schedule parentheses.")
    return entries


class NumericSchedule:
    """Linear numeric anchors; a t-expression is evaluated until the next keyframe."""

    def __init__(self, text: str, *, max_frames: int, fps: float):
        if not isinstance(text, str) or not text.strip() or len(text) > 65536:
            raise ValueError("A numeric schedule requires a non-empty string of at most 64 KiB.")
        self.max_frames, self.fps = max_frames, fps
        self.keyframes: dict[int, Expression] = {}
        if ":" not in text:
            text = "0:(" + text + ")"
        for entry in _entries(text):
            key, separator, value = entry.partition(":")
            if not separator or not value.strip().startswith("(") or not value.strip().endswith(")"):
                raise ValueError("Use the numeric schedule syntax '0:(value), 24:(value)'.")
            expression = Expression(key)
            if expression.dynamic:
                raise ValueError("Frame numbers cannot depend on t.")
            frame = expression.at(0, max_frames - 1, fps)
            if not frame.is_integer() or not 0 <= frame < max_frames or int(frame) in self.keyframes:
                raise ValueError("Schedule frame numbers must be unique integers within the animation.")
            expression = Expression(value.strip()[1:-1])
            expression.at(int(frame), max_frames - 1, fps)
            self.keyframes[int(frame)] = expression
        self.keys = sorted(self.keyframes)

    def at(self, frame: int) -> float:
        if not 0 <= frame < self.max_frames:
            raise ValueError("Frame index is outside the animation.")
        position = max(0, bisect_right(self.keys, frame) - 1)
        left = self.keys[position]
        expression = self.keyframes[left]
        value = expression.at(frame if expression.dynamic else left, self.max_frames - 1, self.fps)
        if expression.dynamic or frame <= left or position == len(self.keys) - 1:
            return value
        right = self.keys[position + 1]
        target = self.keyframes[right].at(right, self.max_frames - 1, self.fps)
        return value + (target - value) * (frame - left) / (right - left)

    def values(self) -> list[float]:
        return [self.at(frame) for frame in range(self.max_frames)]


class PromptSchedule:
    def __init__(self, values: Any, *, max_frames: int):
        if not isinstance(values, dict) or "0" not in values:
            raise ValueError("Prompt schedules require a JSON object containing frame '0'.")
        if any(not isinstance(key, str) or re.fullmatch(r"0|[1-9][0-9]*", key) is None
               or len(key) > 7 or int(key) >= max_frames or not isinstance(value, str)
               for key, value in values.items()):
            raise ValueError("Prompt keys must be canonical frame numbers within the animation, with string values.")
        self.keyframes = {int(key): value for key, value in values.items()}
        self.keys = sorted(self.keyframes)
        self.max_frames = max_frames

    def at(self, frame: int) -> str:
        if not 0 <= frame < self.max_frames:
            raise ValueError("Frame index is outside the animation.")
        return self.keyframes[self.keys[bisect_right(self.keys, frame) - 1]]
