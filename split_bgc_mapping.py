#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Dict


def split_bgc_mapping_by_prefix(mapping_path: str, out_dir: str, prefix_len: int = 4, expected_groups: int = 9) -> Dict[str, int]:
    """
    将BGC映射文件按key的前缀拆分成多个JSON文件。

    参数:
        mapping_path: 输入的bgc_mapping.json路径
        out_dir: 拆分后的输出目录
        prefix_len: 用于分组的前缀长度（默认4）
        expected_groups: 期望的分组数量（默认9，仅用于提示，不做强制校验）

    返回:
        一个字典，key为前缀，value为该组包含的BGC条目数量
    """
    if not os.path.isfile(mapping_path):
        raise FileNotFoundError(f"未找到输入映射文件: {mapping_path}")

    os.makedirs(out_dir, exist_ok=True)

    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    prefix_to_submap: Dict[str, Dict[str, list]] = {}
    for bgc_id, regions in mapping.items():
        prefix = bgc_id[:prefix_len]
        if prefix not in prefix_to_submap:
            prefix_to_submap[prefix] = {}
        prefix_to_submap[prefix][bgc_id] = regions

    for prefix, submap in sorted(prefix_to_submap.items()):
        out_path = os.path.join(out_dir, f"{prefix}.json")
        with open(out_path, 'w') as f:
            json.dump(submap, f, indent=2, ensure_ascii=False)

    group_sizes = {p: len(m) for p, m in prefix_to_submap.items()}

    print("拆分完成:")
    print(f"  输入文件: {mapping_path}")
    print(f"  输出目录: {out_dir}")
    print(f"  前缀长度: {prefix_len}")
    print(f"  分组数: {len(prefix_to_submap)} (期望≈{expected_groups})")
    for p in sorted(group_sizes.keys()):
        print(f"   - {p}: {group_sizes[p]} 条")

    summary_path = os.path.join(out_dir, "split_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "mapping_path": mapping_path,
            "out_dir": out_dir,
            "prefix_len": prefix_len,
            "groups": group_sizes
        }, f, indent=2, ensure_ascii=False)
    print(f"  汇总信息已写入: {summary_path}")

    if len(prefix_to_submap) != expected_groups:
        print("[提示] 实际分组数量与期望不一致，请核对前缀长度或数据。")

    return group_sizes


def main():
    parser = argparse.ArgumentParser("Split BGC mapping JSON by key prefix")
    parser.add_argument("--mapping_json", required=True,
                        help="输入bgc_mapping.json路径")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--prefix_len", type=int,
                        default=4, help="用于分组的前缀长度(默认4)")
    parser.add_argument("--expected_groups", type=int,
                        default=9, help="期望分组数(默认9, 用于提示)")
    args = parser.parse_args()

    split_bgc_mapping_by_prefix(
        mapping_path=args.mapping_json,
        out_dir=args.out_dir,
        prefix_len=args.prefix_len,
        expected_groups=args.expected_groups
    )


if __name__ == "__main__":
    main()
