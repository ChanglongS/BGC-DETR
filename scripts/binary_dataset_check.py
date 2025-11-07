#type: ignore
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小脚本：验证 BGC → 二分类折叠逻辑。

用法示例：
    python scripts/binary_dataset_check.py \
        --mapping_json data/new_data/bgc_mapping.json \
        --sample_limit 5
"""

import argparse
import json
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="Check binary label folding (BGC vs background)")
    parser.add_argument('--mapping_json', required=True, help='路径到 bgc_mapping.json')
    parser.add_argument('--sample_limit', type=int, default=5, help='打印几个样本的折叠结果')
    return parser.parse_args()


def fold_labels_to_binary(mapping: dict):
    """将 mapping 中所有非 no-object 的标签折叠为 'bgc'。"""
    binary_mapping = {}
    for bgc_id, regions in mapping.items():
        folded = []
        for reg in regions:
            label = reg.get('type', '')
            if label == 'no-object':
                folded.append({'type': 'no-object', 'start': reg['start'], 'end': reg['end']})
            else:
                folded.append({'type': 'bgc', 'start': reg['start'], 'end': reg['end']})
        binary_mapping[bgc_id] = folded
    return binary_mapping


def summarize(mapping: dict, title: str):
    label_counter = Counter()
    for regions in mapping.values():
        for reg in regions:
            label_counter[reg.get('type', 'unknown')] += 1
    print(f"\n[{title}] 标签统计：")
    for label, cnt in label_counter.most_common():
        print(f"  {label:12s}: {cnt}")


def main():
    args = parse_args()

    with open(args.mapping_json, 'r') as f:
        mapping = json.load(f)

    summarize(mapping, '原始多分类')

    binary_mapping = fold_labels_to_binary(mapping)
    summarize(binary_mapping, '折叠后二分类')

    print(f"\n示例输出（最多 {args.sample_limit} 个 BGC）：")
    for i, (bgc_id, regions) in enumerate(binary_mapping.items()):
        if i >= args.sample_limit:
            break
        print(f"  > {bgc_id}")
        for reg in regions:
            print(f"      {reg['type']:10s} start={reg['start']:<5} end={reg['end']:<5}")


if __name__ == '__main__':
    main()
