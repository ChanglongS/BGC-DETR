#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import torch
import torch.nn.functional as F
from Bio import SeqIO
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path


def parse_fasta_header(header):
    """
    解析FASTA文件头，提取BGC标号和protein_id

    Args:
        header: FASTA文件头字符串

    Returns:
        tuple: (bgc_id, protein_id, start_pos, end_pos, strand)
    """
    # 格式: >BGC_ID|contig|start-end|strand|protein_id|description|protein_id
    # 示例: >BGC0000001.5|c1|1-1083|+|AEK75490.1|protein_methyltransferase|AEK75490.1

    parts = header.split('|')
    if len(parts) >= 5:
        bgc_id = parts[0]  # BGC0000001.5
        contig = parts[1]  # c1
        position = parts[2]  # 1-1083
        strand = parts[3]  # +
        protein_id = parts[4]  # AEK75490.1

        # 解析位置信息
        start_pos, end_pos = None, None
        if '-' in position:
            start_pos, end_pos = position.split('-')
            start_pos = int(start_pos)
            end_pos = int(end_pos)

        return bgc_id, protein_id, start_pos, end_pos, strand
    else:
        print(f"警告: 无法解析FASTA头: {header}")
        return "unknown", "unknown", None, None, None


def load_esm2_model(model_name="esm2_t33_650M_UR50D"):
    """
    加载ESM2模型

    Args:
        model_name: ESM2模型名称

    Returns:
        model: 加载的ESM2模型
        alphabet: 模型的字母表
    """
    print(f"正在加载ESM2模型: {model_name}")

    try:
        import esm
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        model.eval()
        print(f"ESM2模型加载成功: {model_name}")
        return model, alphabet
    except ImportError:
        print("错误: 请先安装ESM库: pip install fair-esm")
        return None, None
    except Exception as e:
        print(f"错误: 无法加载ESM2模型: {e}")
        return None, None


def extract_embeddings(model, alphabet, sequences, device='cuda'):
    """
    使用ESM2模型提取序列嵌入

    Args:
        model: ESM2模型
        alphabet: 模型的字母表
        sequences: 序列列表 [(seq_id, sequence), ...]
        device: 计算设备

    Returns:
        dict: {seq_id: embedding} 嵌入字典
    """
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    embeddings = {}

    with torch.no_grad():
        for seq_id, seq in tqdm(sequences, desc="提取嵌入"):
            try:
                # 准备批次数据
                batch_labels, batch_strs, batch_tokens = batch_converter([
                                                                         (seq_id, seq)])
                batch_tokens = batch_tokens.to(device)

                # 模型前向传播
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33])  # 使用第33层表示

                # 提取嵌入 (去除CLS和EOS token)
                # [seq_len, hidden_dim]
                token_embeddings = results["representations"][33][0]

                # 计算序列级别的嵌入 (平均池化)
                sequence_embedding = token_embeddings.mean(dim=0).cpu().numpy()

                embeddings[seq_id] = sequence_embedding

            except Exception as e:
                print(f"警告: 处理序列 {seq_id} 时出错: {e}")
                # 如果出错，使用零向量
                hidden_dim = model.args.embed_dim
                embeddings[seq_id] = np.zeros(hidden_dim)

    return embeddings


def process_fasta_file(fasta_file, output_dir, model_name="esm2_t33_650M_UR50D", device='cuda'):
    """
    处理FASTA文件，提取每个CDS的嵌入

    Args:
        fasta_file: FASTA文件路径
        output_dir: 输出目录
        model_name: ESM2模型名称
        device: 计算设备
    """
    print(f"正在处理FASTA文件: {fasta_file}")
    print(f"输出目录: {output_dir}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载ESM2模型
    model, alphabet = load_esm2_model(model_name)
    if model is None:
        return

    # 读取FASTA文件
    sequences = []
    bgc_info = {}

    print("正在读取FASTA文件...")
    for record in SeqIO.parse(fasta_file, "fasta"):
        bgc_id, protein_id, start_pos, end_pos, strand = parse_fasta_header(
            record.id)

        # 存储序列信息
        seq_id = f"{bgc_id}_{protein_id}"
        sequences.append((seq_id, str(record.seq)))

        # 存储BGC信息
        if bgc_id not in bgc_info:
            bgc_info[bgc_id] = []

        bgc_info[bgc_id].append({
            'protein_id': protein_id,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'strand': strand,
            'sequence': str(record.seq),
            'seq_id': seq_id
        })

    print(f"共读取到 {len(sequences)} 个序列")
    print(f"来自 {len(bgc_info)} 个BGC")

    # 提取嵌入
    print("正在提取序列嵌入...")
    embeddings = extract_embeddings(model, alphabet, sequences, device)

    # 保存嵌入文件
    print("正在保存嵌入文件...")

    # 按BGC分组保存
    for bgc_id, proteins in bgc_info.items():
        bgc_embeddings = {}

        for protein_info in proteins:
            seq_id = protein_info['seq_id']
            protein_id = protein_info['protein_id']

            # 获取对应的嵌入
            if seq_id in embeddings:
                bgc_embeddings[protein_id] = embeddings[seq_id]
            else:
                print(f"警告: 未找到序列 {seq_id} 的嵌入")

        # 保存BGC的嵌入文件
        bgc_output_file = os.path.join(output_dir, f"{bgc_id}_embeddings.npz")
        np.savez_compressed(
            bgc_output_file,
            **{protein_id: embedding for protein_id, embedding in bgc_embeddings.items()}
        )

        print(f"已保存 {bgc_id} 的嵌入文件: {bgc_output_file}")
        print(f"  包含 {len(bgc_embeddings)} 个蛋白质的嵌入")

    # 保存元数据
    metadata_file = os.path.join(output_dir, "metadata.json")
    metadata = {
        'fasta_file': fasta_file,
        'model_name': model_name,
        'total_sequences': len(sequences),
        'total_bgcs': len(bgc_info),
        'bgc_info': bgc_info
    }

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"元数据已保存到: {metadata_file}")
    print("处理完成!")


def main():
    parser = argparse.ArgumentParser(description='FASTA文件ESM2嵌入提取工具')

    parser.add_argument('--fasta_file', required=True, help='输入的FASTA文件路径')
    parser.add_argument('--output_dir', required=True, help='输出目录路径')
    parser.add_argument('--model_name', default='esm2_t33_650M_UR50D',
                        help='ESM2模型名称 (默认: esm2_t33_650M_UR50D)')
    parser.add_argument('--device', default='cuda', help='计算设备 (默认: cuda)')

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.fasta_file):
        print(f"错误: 输入文件不存在: {args.fasta_file}")
        return

    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，切换到CPU")
        args.device = 'cpu'

    print("=== FASTA文件ESM2嵌入提取工具 ===")
    print(f"输入文件: {args.fasta_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"模型名称: {args.model_name}")
    print(f"计算设备: {args.device}")

    # 处理FASTA文件
    process_fasta_file(
        fasta_file=args.fasta_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device
    )


if __name__ == '__main__':
    main()


# 使用示例:
# python fasta_to_embeddings_simple.py --fasta_file proteins.fasta --output_dir embeddings/
#
# 输出文件结构:
# embeddings/
# ├── BGC0000001.5_embeddings.npz    # BGC的嵌入文件
# ├── BGC0000002.1_embeddings.npz    # 另一个BGC的嵌入文件
# └── metadata.json                   # 元数据文件
#
# 加载嵌入文件:
# import numpy as np
# data = np.load('embeddings/BGC0000001.5_embeddings.npz')
# embedding = data['AEK75490.1']  # 获取特定蛋白质的嵌入
#
# 查看嵌入维度:
# print(f"嵌入维度: {embedding.shape}")  # 应该是 (1280,) 对于esm2_t33_650M_UR50D
