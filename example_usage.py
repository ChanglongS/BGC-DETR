#!/usr/bin/env python3
import os
import re
import argparse
import json
import sqlite3
import mmap
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from esm import pretrained, Alphabet
from esm.data import FastaBatchedDataset
import re
from torch.utils.data import DataLoader


def create_sequence_db(cds_sequences, db_path):
    """将序列数据存入 SQLite 数据库"""
    # 检查数据库是否已存在
    if os.path.exists(db_path):
        print("序列数据库已存在，跳过创建")
        return

    print("正在创建序列数据库...")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sequences
                 (protein_id TEXT PRIMARY KEY, sequence TEXT)''')

    # 批量插入数据
    c.executemany('INSERT OR REPLACE INTO sequences VALUES (?, ?)',
                  cds_sequences.items())
    conn.commit()
    conn.close()
    print("序列数据库创建完成")


def get_sequence_from_db(protein_id, db_path):
    """从数据库获取序列"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT sequence FROM sequences WHERE protein_id = ?', (protein_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None


def chunk_and_embed(tokens, model, proj, device, max_len, layer, padding_idx):
    B, T = tokens.shape
    assert B == 1, "batch_size must be 1 for chunked embedding"
    chunks = []
    for start in range(0, T, max_len):
        end = min(start + max_len, T)
        sub = tokens[:, start:end].to(device)
        clen = end - start
        if clen < max_len:
            pad = torch.full(
                (1, max_len - clen), fill_value=padding_idx,
                dtype=torch.long, device=device
            )
            sub = torch.cat([sub, pad], dim=1)

        out = model(sub, repr_layers=[layer], return_contacts=False)
        rep = out["representations"][layer]       # [1, max_len, 1280]
        x = rep.permute(0, 2, 1)                  # [1,1280,max_len]
        x = proj(x)                               # [1,256,max_len]
        x = x.permute(0, 2, 1).cpu().detach().numpy()      # [1,max_len,256]
        chunks.append(x[0, :clen])

    return np.concatenate(chunks, axis=0)       # [T,256]


def load_json(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def load_sequences_mmap(json_file):
    """使用内存映射加载大文件"""
    with open(json_file, 'r') as f:
        # 创建内存映射
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # 使用 json 解析器
        return json.loads(mm.read().decode('utf-8'))


def generate_embedding_batch(sequences, model, alphabet, proj, device, max_len=1024, layer=-1):
    """批量生成序列的嵌入"""
    def sanitize(seq: str) -> str:
        # 移除空白并大写
        s = re.sub(r"\s+", "", seq).upper()
        # 将不在ESM字母表中的字符统一替换为X
        # 允许的标准氨基酸字母（不含B/Z/J/U/O），其余替换
        allowed = set(list("ACDEFGHIKLMNPQRSTVWY"))
        s = "".join(ch if ch in allowed else "X" for ch in s)
        return s
    # 1. 将所有序列清洗并转换为 tokens
    batch_tokens = []
    for sequence in sequences:
        sequence = sanitize(sequence)
        tokens = alphabet.encode(sequence)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        batch_tokens.append(tokens)

    # 2. 找到最长的序列长度
    max_len_batch = max(len(tokens) for tokens in batch_tokens)

    # 3. 填充所有序列到相同长度
    padded_tokens = []
    for tokens in batch_tokens:
        if len(tokens) < max_len_batch:
            pad = [alphabet.padding_idx] * (max_len_batch - len(tokens))
            tokens = tokens + pad
        padded_tokens.append(tokens)

    # 4. 转换为张量
    batch_tokens = torch.tensor(padded_tokens, dtype=torch.long, device=device)

    # 5. 使用模型生成嵌入
    with torch.no_grad():
        out = model(batch_tokens, repr_layers=[layer], return_contacts=False)
        rep = out["representations"][layer]  # [batch_size, seq_len, 1280]
        x = rep.permute(0, 2, 1)  # [batch_size, 1280, seq_len]
        x = proj(x)  # [batch_size, 256, seq_len]
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, 256]
        return x.cpu().numpy()


def process_chunk(chunk_data, device_id, model_name, layer, output_dir, db_path, cds_sequences_file, force_reprocess=False, device_type="cuda"):
    """处理数据块的函数，直接输出簇级嵌入，每个BGC只保存一个npy文件，使用mean pooling"""
    if device_type == "cpu":
        device = "cpu"
    else:
        device = f"cuda:{device_id}"

    # 加载模型和字母表
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.to(device).eval()

    # 投影层
    proj = torch.nn.Conv1d(1280, 256, kernel_size=1).to(device)

    total = len(chunk_data)
    processed = 0
    skipped = 0
    failed = 0

    # 批处理设置
    BATCH_SIZE = 128  # 可以根据GPU内存调整

    # 只保留一个总体进度条
    if device_type == "cpu":
        desc = f"CPU BGC处理进度"
    else:
        desc = f"GPU {device_id} BGC处理进度"

    for bgc_id, protein_ids in tqdm(chunk_data, desc=desc, total=total):
        out_path = os.path.join(output_dir, f"{bgc_id}.npy")
        if os.path.exists(out_path) and not force_reprocess:
            skipped += 1
            continue

        prot_vecs = []
        success_count = 0

        # 批量获取序列
        sequences = []
        valid_protein_ids = []
        for protein_id in protein_ids:
            sequence = get_sequence_from_db(protein_id, db_path)
            if not sequence:
                try:
                    with open(cds_sequences_file, 'r') as f:
                        cds_data = json.load(f)
                        sequence = cds_data.get(protein_id)
                    if sequence:
                        print(f"从原始JSON文件中找到 {protein_id} 的序列")
                except Exception as e:
                    print(f"! 无法从JSON文件读取 {protein_id} 的序列: {str(e)}")

            if sequence:
                sequences.append(sequence)
                valid_protein_ids.append(protein_id)

        if not sequences:
            print(f"! BGC-{bgc_id} 没有找到任何有效序列")
            continue

        # 批量处理序列
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_sequences = sequences[i:i + BATCH_SIZE]
            batch_protein_ids = valid_protein_ids[i:i + BATCH_SIZE]

            try:
                # 批量生成嵌入
                embeddings = generate_embedding_batch(
                    batch_sequences,
                    model,
                    alphabet,
                    proj,
                    device,
                    max_len=1024,
                    layer=layer
                )

                # 计算每个序列的平均嵌入
                for emb in embeddings:
                    vec = emb.mean(axis=0)
                    prot_vecs.append(vec)
                success_count += len(batch_sequences)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    # 如果内存不足，减小批处理大小重试
                    print(f"内存不足，减小批处理大小重试: {str(e)}")
                    BATCH_SIZE = max(1, BATCH_SIZE // 2)
                    continue
                print(f"! BGC-{bgc_id} 的批处理失败: {str(e)}")
                failed += len(batch_protein_ids)
            except Exception as e:
                print(f"! BGC-{bgc_id} 的批处理失败: {str(e)}")
                failed += len(batch_protein_ids)

        if prot_vecs:
            cluster_emb = np.stack(prot_vecs, axis=0)
            np.save(out_path, cluster_emb)
            processed += 1
        else:
            print(f"! BGC-{bgc_id} 处理失败：没有成功处理任何蛋白质")

    if device_type == "cpu":
        print(f"\nCPU 处理完成:")
    else:
        print(f"\nGPU {device_id} 处理完成:")
    print(f"- 成功处理: {processed} 个BGC")
    print(f"- 已跳过: {skipped} 个BGC")
    print(f"- 处理失败: {failed} 个蛋白质序列")

    return {
        "total": total,
        "processed": processed,
        "skipped": skipped,
        "failed": failed
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for CDS sequences")
    parser.add_argument("--bgc_proteins", required=True,
                        help="BGC proteins JSON file")
    parser.add_argument("--cds_sequences", required=True,
                        help="CDS sequences JSON file")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for embeddings")
    parser.add_argument(
        "--model", default="esm2_t33_650M_UR50D", help="ESM2 model name")
    parser.add_argument("--layer", type=int, default=33,
                        help="Layer to extract embeddings from")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="要使用的GPU ID，用逗号分隔，例如：0,5")
    parser.add_argument("--force", action="store_true",
                        help="强制重新处理所有序列，即使已经存在")
    args = parser.parse_args()

    # 1. 加载 BGC proteins 和 CDS sequences
    bgc_proteins = load_json(args.bgc_proteins)
    cds_sequences = load_sequences_mmap(args.cds_sequences)

    # 2. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. 创建序列数据库
    db_path = os.path.join(args.output_dir, "sequences.db")
    create_sequence_db(cds_sequences, db_path)

    # 4. 设备选择逻辑
    if args.device == "cpu":
        # CPU模式：使用单进程处理
        device_ids = ["cpu"]
        num_devices = 1
        print(f"使用设备: {args.device}")
    else:
        # GPU模式：解析GPU ID
        device_ids = [int(id.strip()) for id in args.gpu_ids.split(',')]
        num_devices = len(device_ids)
        print(f"使用GPU: {device_ids}")

    # 5. 将数据分成多个块
    bgc_items = list(bgc_proteins.items())
    chunk_size = len(bgc_items) // num_devices
    chunks = [bgc_items[i:i + chunk_size]
              for i in range(0, len(bgc_items), chunk_size)]

    # 6. 使用多进程处理
    processes = []
    for i, device_id in enumerate(device_ids):
        p = mp.Process(
            target=process_chunk,
            args=(chunks[i], device_id, args.model, args.layer, args.output_dir,
                  db_path, args.cds_sequences, args.force, args.device)
        )
        p.start()
        processes.append(p)

    # 7. 等待所有进程完成
    for p in processes:
        p.join()

    # 8. 处理完成
    print("\n 所有BGC嵌入处理完成！")
    print(f"总BGC数量：{len(bgc_items)}")
    if args.device == "cpu":
        print(f"使用设备：{args.device}")
    else:
        print(f"使用GPU：{device_ids}")
    print(f"输出目录：{args.output_dir}")

    # 检查输出文件数量
    import glob
    npy_files = glob.glob(os.path.join(args.output_dir, "*.npy"))
    print(f"生成的嵌入文件数量：{len(npy_files)}")


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()

# 使用示例：
# 1. 使用CPU，自动跳过已处理的文件（断点续传）：
# python scripts/example_usage.py --bgc_proteins data/val/bgc_proteins.json --cds_sequences data/val/cds_sequences.json --output_dir data/val/embeddings --model esm2_t33_650M_UR50D --layer 33 --device cpu
#
# 2. 强制重新处理所有文件（不使用断点续传）：
# python scripts/example_usage.py --bgc_proteins data/val/bgc_proteins.json --cds_sequences data/val/cds_sequences.json --output_dir data/val/embeddings --model esm2_t33_650M_UR50D --layer 33 --device cuda --gpu_ids 2 --force
#
# 3. 如果之后可以使用更多GPU，只需要修改 --gpu_ids 参数：
# python scripts/example_usage.py --bgc_proteins data/bgc_proteins.json --cds_sequences data/cds_sequences.json --output_dir embeddings --model esm2_t33_650M_UR50D --layer 33 --device cuda --gpu_ids 0,1,2,3
