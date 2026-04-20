from __future__ import annotations

import argparse
import sys


class ChineseArgumentParser(argparse.ArgumentParser):
    def format_usage(self) -> str:
        return _translate(super().format_usage())

    def format_help(self) -> str:
        return _translate(super().format_help())

    def exit(self, status: int = 0, message: str | None = None) -> None:
        super().exit(status, _translate(message) if message else None)

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: 错误：{_translate(message)}\n")


def _translate(text: str) -> str:
    return (
        text.replace("usage:", "用法：")
        .replace("options:", "参数：")
        .replace("optional arguments:", "参数：")
        .replace("show this help message and exit", "显示帮助信息并退出")
        .replace("the following arguments are required:", "以下参数是必填项：")
        .replace("unrecognized arguments:", "无法识别的参数：")
        .replace("expected one argument", "需要一个参数值")
        .replace("invalid int value:", "无效的整数值：")
        .replace("invalid float value:", "无效的浮点数值：")
    )
