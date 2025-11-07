#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理待预测数据脚本（修复版）
将proteins文件夹中的FASTA文件转换成BGC-DETR预测所需的格式
"""

import os
import json
import argparse
import glob
from Bio import SeqIO
from tqdm import tqdm
import numpy as np


def parse_fasta_files(proteins_dir, target_length=128):
    """
    解析proteins文件夹中的FASTA文件，提取序列信息

    Args:
        proteins_dir: proteins文件夹路径
        target_length: 目标序列长度（CDS数量）

    Returns:
        tuple: (bgc_mapping, bgc_proteins, cds_sequences)
    """
    bgc_mapping = {}
    bgc_proteins = {}
    cds_sequences = {}

    fasta_pattern = os.path.join(proteins_dir, "*.fa")
    fasta_files = glob.glob(fasta_pattern)

    print("找到 {} 个FASTA文件".format(len(fasta_files)))

    for fasta_file in tqdm(fasta_files, desc="处理FASTA文件"):
        filename = os.path.basename(fasta_file)
        genome_id = filename.replace(
            'genome.protein_', 'PRED_').replace('.fa', '')

        # 读取序列
        sequences = []
        protein_ids = []

        for record in SeqIO.parse(fasta_file, "fasta"):
            # 序列ID格式: >101_1_1 # 609 # 1187 # 1 # ID=1_1;...
            protein_id = "{}_{}".format(genome_id, record.id)
            sequences.append(str(record.seq))
            protein_ids.append(protein_id)
            cds_sequences[protein_id] = str(record.seq)

        # 将序列分割成固定长度的片段进行预测
        num_sequences = len(sequences)
        print("基因组 {}: {} 个蛋白质序列".format(genome_id, num_sequences))

        # 分割成多个片段，每个片段最多target_length个蛋白质
        for i in range(0, num_sequences, target_length):
            segment_end = min(i + target_length, num_sequences)
            segment_protein_ids = protein_ids[i:segment_end]

            # 只有当序列数量足够时才创建片段（避免过多的padding）
            if len(segment_protein_ids) >= 10:  # 至少要有10个蛋白质才值得预测
                # 如果片段长度不足target_length，创建padding蛋白质
                if len(segment_protein_ids) < target_length:
                    padding_needed = target_length - len(segment_protein_ids)
                    for j in range(padding_needed):
                        padding_id = "{}_padding_{}".format(genome_id, j)
                        if padding_id not in cds_sequences:
                            # 创建一个简单的padding蛋白质序列
                            cds_sequences[padding_id] = "M" * 50  # 50个氨基酸的简单序列
                        segment_protein_ids.append(padding_id)

                # 创建段落ID
                segment_id = "{}_seg_{}".format(genome_id, i//target_length)

                # 生成映射信息（预测模式，不包含真实标签）
                bgc_mapping[segment_id] = [{
                    "start": 0,
                    "end": len(segment_protein_ids) - 1,
                    "type": "unknown"  # 待预测，类型未知
                }]

                bgc_proteins[segment_id] = segment_protein_ids
            else:
                print("跳过片段（蛋白质数量不足）: {} 个蛋白质".format(len(segment_protein_ids)))

    return bgc_mapping, bgc_proteins, cds_sequences


def save_prediction_data(bgc_mapping, bgc_proteins, cds_sequences, output_dir):
    """
    保存预测数据文件

    Args:
        bgc_mapping: BGC映射信息
        bgc_proteins: BGC蛋白质列表
        cds_sequences: CDS序列字典
        output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存映射文件
    mapping_file = os.path.join(output_dir, "bgc_mapping.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(bgc_mapping, f, ensure_ascii=False, indent=2)
    print("保存映射文件: {}".format(mapping_file))

    # 保存蛋白质文件
    proteins_file = os.path.join(output_dir, "bgc_proteins.json")
    with open(proteins_file, 'w', encoding='utf-8') as f:
        json.dump(bgc_proteins, f, ensure_ascii=False, indent=2)
    print("保存蛋白质文件: {}".format(proteins_file))

    # 保存序列文件
    sequences_file = os.path.join(output_dir, "cds_sequences.json")
    with open(sequences_file, 'w', encoding='utf-8') as f:
        json.dump(cds_sequences, f, ensure_ascii=False, indent=2)
    print("保存序列文件: {}".format(sequences_file))

    # 输出统计信息
    print("\n数据统计:")
    print("- 预测片段数量: {}".format(len(bgc_mapping)))
    print("- 总蛋白质数量: {}".format(len(cds_sequences)))
    print("- 输出目录: {}".format(output_dir))


def main():
    parser = argparse.ArgumentParser(description="处理待预测数据")
    parser.add_argument("--proteins_dir", type=str, required=True,
                        help="proteins文件夹路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--target_length", type=int, default=128,
                        help="目标序列长度")

    args = parser.parse_args()

    print("开始处理待预测数据...")

    # 解析FASTA文件
    bgc_mapping, bgc_proteins, cds_sequences = parse_fasta_files(
        args.proteins_dir, args.target_length
    )

    # 保存数据
    save_prediction_data(bgc_mapping, bgc_proteins,
                         cds_sequences, args.output_dir)

    print("数据处理完成！")


if __name__ == "__main__":
    main()
