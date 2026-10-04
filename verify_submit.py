#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCKS2026 大模型生成文本检测及溯源 —— 提交文件校验脚本
=====================================================

检查 submit.jsonl 的格式合法性：
  - label 取值 ∈ {0, 1, 2}
  - family 取值 ∈ {-1, 0, 1, ..., 7}（-1 表示非纯机器文本，无需溯源）

用法示例：
  python verify_submit.py --submit submit.jsonl
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='CCKS2026 AI 文本检测与溯源 - 提交校验')
    parser.add_argument('--submit', default='submit.jsonl', help='提交文件路径')
    args = parser.parse_args()

    with open(args.submit, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    total = len(lines)
    print(f"Total: {total}")

    first = json.loads(lines[0])
    last = json.loads(lines[-1])
    print(f"First: id={first['id']}, label={first['label']}, family={first['family']}")
    print(f"Last: id={last['id']}, label={last['label']}, family={last['family']}")

    ok = True
    for i, line in enumerate(lines):
        try:
            d = json.loads(line)
            if d['label'] not in (0, 1, 2) or d['family'] not in (-1, 0, 1, 2, 3, 4, 5, 6, 7):
                print(f"Invalid at line {i}: {d}")
                ok = False
        except Exception as e:
            print(f"Error line {i}: {e}")
            ok = False
    if ok:
        print("All records valid!")


if __name__ == '__main__':
    main()
