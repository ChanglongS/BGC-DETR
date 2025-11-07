#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Dict, Tuple


def extract_auc(metrics: dict) -> float:
    # 优先使用CDS级别AUC
    if isinstance(metrics, dict):
        cds = metrics.get('cds_level_metrics', {})
        if isinstance(cds, dict) and isinstance(cds.get('auc', None), (int, float)):
            return float(cds['auc'])
        # 备选：顶层localization_auc
        if isinstance(metrics.get('localization_auc', None), (int, float)):
            return float(metrics['localization_auc'])
        # 备选：test_metrics结构中的 localization.auc
        loc = metrics.get('localization', {})
        if isinstance(loc, dict) and isinstance(loc.get('auc', None), (int, float)):
            return float(loc['auc'])
    return 0.0


def find_best_auc_per_genome(results_dir: str) -> Dict[str, Tuple[str, float]]:
    """
    遍历results_dir下的模型子目录，读取每个基因组的*_metrics.json，
    取每个基因组AUC的最大值与对应模型名。
    返回: {genome_name: (model_name, best_auc)}
    """
    best: Dict[str, Tuple[str, float]] = {}

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"结果目录不存在: {results_dir}")

    for model_name in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        # 跳过内部目录名
        for fn in sorted(os.listdir(model_dir)):
            if not fn.endswith('_metrics.json'):
                continue
            genome_name = fn[:-len('_metrics.json')]
            metrics_path = os.path.join(model_dir, fn)
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                auc = extract_auc(metrics)
            except Exception as e:
                print(f"读取失败: {metrics_path}: {e}")
                continue

            if genome_name not in best or auc > best[genome_name][1]:
                best[genome_name] = (model_name, auc)

    return best


def main():
    parser = argparse.ArgumentParser('Print best AUC per genome across models')
    parser.add_argument(
        '--results_dir', default='./my_validation_results', help='验证输出顶层目录')
    args = parser.parse_args()

    best = find_best_auc_per_genome(args.results_dir)
    if not best:
        print('未在指定目录下找到任何 *_metrics.json 文件。')
        return

    print('\n每个基因组的最佳AUC对应模型:')
    for genome in sorted(best.keys()):
        model_name, auc = best[genome]
        print(f"  {genome}: {auc:.4f} @ {model_name}")


if __name__ == '__main__':
    main()
