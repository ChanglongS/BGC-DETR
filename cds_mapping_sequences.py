import re
import json
import os
import argparse
from tqdm import tqdm
from Bio import SeqIO


def extract_cds_from_file(gbk_file):
    """
    只提取特征名严格为 CDS 的 protein_id/locus_tag/gene 和 translation
    """
    cds_dict = {}
    try:
        with open(gbk_file, encoding="utf-8") as f:
            lines = f.readlines()
        in_cds = False
        cds_lines = []
        for line in lines:
            m = re.match(r'^\s{5}([a-zA-Z_]+)\s', line)
            if m:
                feature_type = m.group(1)
                if in_cds:
                    # 处理上一个CDS块
                    cds_text = ''.join(cds_lines)
                    pid_match = re.search(r'/protein_id="([^"]+)"', cds_text)
                    locus_match = re.search(r'/locus_tag="([^"]+)"', cds_text)
                    gene_match = re.search(r'/gene="([^"]+)"', cds_text)
                    # translation多行拼接
                    translation = None
                    t_start = False
                    t_lines = []
                    for l in cds_text.splitlines():
                        if '/translation="' in l:
                            t_start = True
                            t_lines.append(l.split('/translation="', 1)[1])
                            if l.rstrip().endswith('"') and l.count('"') == 2:
                                t_start = False
                                break
                            continue
                        if t_start:
                            t_lines.append(l)
                            if l.rstrip().endswith('"'):
                                t_start = False
                                break
                    if t_lines:
                        t_str = '\n'.join(t_lines)
                        if t_str.endswith('"'):
                            t_str = t_str[:-1]
                        translation = re.sub(r'\s+', '', t_str)
                    seq_id = (
                        pid_match.group(1) if pid_match else
                        (locus_match.group(1) if locus_match else
                         (gene_match.group(1) if gene_match else None))
                    )
                    if seq_id and translation:
                        cds_dict[seq_id] = translation
                    cds_lines = []
                    in_cds = False
                # 只在严格等于 'CDS' 时才进入
                if feature_type == 'CDS':
                    in_cds = True
                    cds_lines = [line]
            elif in_cds:
                cds_lines.append(line)
        # 处理最后一个CDS块
        if in_cds and cds_lines:
            cds_text = ''.join(cds_lines)
            pid_match = re.search(r'/protein_id="([^"]+)"', cds_text)
            locus_match = re.search(r'/locus_tag="([^"]+)"', cds_text)
            gene_match = re.search(r'/gene="([^"]+)"', cds_text)
            translation = None
            t_start = False
            t_lines = []
            for l in cds_text.splitlines():
                if '/translation="' in l:
                    t_start = True
                    t_lines.append(l.split('/translation="', 1)[1])
                    if l.rstrip().endswith('"') and l.count('"') == 2:
                        t_start = False
                        break
                    continue
                if t_start:
                    t_lines.append(l)
                    if l.rstrip().endswith('"'):
                        t_start = False
                        break
            if t_lines:
                t_str = '\n'.join(t_lines)
                if t_str.endswith('"'):
                    t_str = t_str[:-1]
                translation = re.sub(r'\s+', '', t_str)
            seq_id = (
                pid_match.group(1) if pid_match else
                (locus_match.group(1) if locus_match else
                 (gene_match.group(1) if gene_match else None))
            )
            if seq_id and translation:
                cds_dict[seq_id] = translation
    except Exception as e:
        print(f"正则法处理文件 {os.path.basename(gbk_file)} 时出错: {e}")
    return cds_dict


def extract_single_gbk(gbk_file, out_json=None):
    print(f"\n处理文件: {os.path.basename(gbk_file)}")
    cds_dict = extract_cds_from_file(gbk_file)
    if not cds_dict:
        print(f"警告: 文件 {os.path.basename(gbk_file)} 未能提取到任何 CDS！")
    if out_json:
        with open(out_json, 'w') as f:
            json.dump(cds_dict, f, indent=2)
    return cds_dict


def extract_cds_batch(gbk_dir, out_json):
    cds_dict = {}
    files = [f for f in os.listdir(
        gbk_dir) if f.lower().endswith(('.gbk', '.gbff'))]
    print(f"找到 {len(files)} 个 GBK/GBFF 文件")
    for fn in tqdm(sorted(files), desc="提取 CDS 序列"):
        path = os.path.join(gbk_dir, fn)
        sub_dict = extract_cds_from_file(path)
        cds_dict.update(sub_dict)
    print(f"\n共提取到 {len(cds_dict)} 条 CDS 序列")
    with open(out_json, "w") as fw:
        json.dump(cds_dict, fw, indent=2)
    print(f"结果已保存到 {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取 GBK 文件中的 CDS translation")
    parser.add_argument("--gbk_dir", help="GBK 目录，用于批量处理")
    parser.add_argument("--single_gbk", help="单个 GBK 文件路径")
    parser.add_argument("--out_json", required=True, help="输出 JSON 路径")
    args = parser.parse_args()
    if args.gbk_dir:
        extract_cds_batch(args.gbk_dir, args.out_json)
    elif args.single_gbk:
        extract_single_gbk(args.single_gbk, args.out_json)
    else:
        parser.error("请指定 --gbk_dir 或 --single_gbk")


#   python scripts/cds_mapping_sequences.py --single_gbk data/genomes/BGC0001317.gbk --out_json cds_sequences.json
#   python scripts/cds_mapping_sequences.py --gbk_dir data/nine_genomes_gbff --out_json data/val/cds_sequences.json
