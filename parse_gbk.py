#!/usr/bin/env python3
import os
import re
import json
import argparse
import random
from Bio import SeqIO
from pathlib import Path
from tqdm import tqdm

# 7 类映射函数


def map_to_7class(lbl: str) -> str:
    L = lbl.lower()
    # 直接匹配antismash格式的类别名称
    if L == "nrps":
        return "NRPS"
    if L == "pks":
        return "PKS"
    if L == "ripp":
        return "RiPP"
    if L == "terpene":
        return "Terpene"
    if L == "saccharide":
        return "Saccharide"
    if L == "alkaloid":
        return "Alkaloid"
    # 兼容传统格式的模糊匹配
    if "nrps" in L and "pks" not in L:
        return "NRPS"
    if "pks" in L and "nrps" not in L:
        return "PKS"
    if "ripp" in L or "ripe" in L:
        return "RiPP"
    if "saccharide" in L or "glyco" in L:
        return "Saccharide"
    if "terpene" in L:
        return "Terpene"
    if "alkaloid" in L or "alkalo" in L:
        return "Alkaloid"
    return "Other"


def find_cds_index(pos, cds_positions):
    for i, (start, end) in enumerate(cds_positions):
        if start <= pos <= end:
            return i
    # 如果没找到，返回最接近的
    if not cds_positions:
        return None
    if pos < cds_positions[0][0]:
        return 0
    if pos > cds_positions[-1][1]:
        return len(cds_positions) - 1
    closest = min(range(len(cds_positions)), key=lambda i: min(
        abs(cds_positions[i][0] - pos), abs(cds_positions[i][1] - pos)))
    return closest


def get_cds_positions(gbk_file):
    cds_positions = []
    try:
        for record in SeqIO.parse(gbk_file, "genbank"):
            for feature in record.features:
                if feature.type == "CDS":
                    if feature.location.__class__.__name__ == "CompoundLocation":
                        starts = [int(part.start)
                                  for part in feature.location.parts]
                        ends = [int(part.end)
                                for part in feature.location.parts]
                        cds_positions.append((min(starts), max(ends)))
                    else:
                        start = int(feature.location.start)
                        end = int(feature.location.end)
                        cds_positions.append((start, end))
    except Exception as e:
        print(f"[get_cds_positions] WARNING: 解析 {gbk_file} 失败，已跳过。原因：{e}")
    return sorted(cds_positions)


def parse_gbk_text(gbk_file, bgc_id=None):
    cds_positions = []
    subregions = []
    protein_ids = []
    with open(gbk_file) as f:
        lines = f.readlines()

    # 1. 提取所有CDS区间和蛋白ID
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^\s{5}CDS\s+(.+)', line)
        if m:
            loc = m.group(1).strip()
            nums = re.findall(r'(\d+)\.\.(\d+)', loc)
            if nums:
                starts = [int(x[0]) for x in nums]
                ends = [int(x[1]) for x in nums]
                cds_positions.append((min(starts), max(ends)))
            # 向下查找 protein_id/locus_tag/gene
            pid, locus, gene = None, None, None
            j = i + 1
            while j < len(lines) and not re.match(r'^\s{5}\S', lines[j]):
                if '/protein_id=' in lines[j]:
                    pid_match = re.search(
                        r'/protein_id="?([^"\s]+)"?', lines[j])
                    if pid_match:
                        pid = pid_match.group(1)
                if '/locus_tag=' in lines[j]:
                    locus_match = re.search(
                        r'/locus_tag="?([^"\s]+)"?', lines[j])
                    if locus_match:
                        locus = locus_match.group(1)
                if '/gene=' in lines[j]:
                    gene_match = re.search(r'/gene="?([^"\s]+)"?', lines[j])
                    if gene_match:
                        gene = gene_match.group(1)
                j += 1
            if pid:
                protein_ids.append(pid)
            elif locus:
                protein_ids.append(locus)
            elif gene:
                protein_ids.append(gene)
            else:
                protein_ids.append("unknown")
            i = j - 1
        i += 1

    # 2. 提取所有subregion区间和label（支持传统格式）
    cur = None
    for line in lines:
        m = re.match(r'^\s*subregion\s+(\d+)\.\.(\d+)', line)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            cur = {"start": start, "end": end, "type": None}
            continue
        m2 = re.match(r'^\s*/label="(.+)"', line)
        if cur and m2:
            cur["type"] = m2.group(1)
            subregions.append(cur)
            cur = None

    # 3. 提取所有protocluster区间和category（支持antismash格式）
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^\s{5}protocluster\s+(\d+)\.\.(\d+)', line)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            cur_protocluster = {"start": start, "end": end, "type": None}
            # 向下查找category字段
            j = i + 1
            while j < len(lines) and not re.match(r'^\s{5}\S', lines[j]):
                if '/category=' in lines[j]:
                    category_match = re.search(r'/category="([^"]+)"', lines[j])
                    if category_match:
                        cur_protocluster["type"] = category_match.group(1)
                        break
                j += 1
            if cur_protocluster["type"]:
                subregions.append(cur_protocluster)
            i = j - 1
        i += 1

    if bgc_id is not None:
        print(f"{bgc_id} CDS positions: {cds_positions}")
    print(f"{bgc_id} subregions: {subregions}")
    return cds_positions, subregions, protein_ids


def load_non_bgc_cds(output_dir):
    """递归加载所有非 BGC CDS 信息，只保留有 protein_id 的"""
    cds_list = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith("_non_bgc_cds.json"):
                json_path = os.path.join(root, file)
                try:
                    with open(json_path) as f:
                        data = json.load(f)
                        # 只保留有 protein_id 的
                        data = [
                            cds for cds in data if "protein_id" in cds and cds["protein_id"]]
                        cds_list.extend(data)
                        print(
                            f"Loaded {len(data)} CDS with protein_id from {json_path}")
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")
    if not cds_list:
        print(f"[load_non_bgc_cds] 未找到非BGC CDS json文件")
    else:
        print(
            f"[load_non_bgc_cds] 共加载 {len(cds_list)} 个非BGC CDS（有 protein_id）")
    return cds_list


def load_non_bgc_protein_ids(output_dir):
    """递归加载所有非 BGC CDS 的 protein_id，去除空字符串和重复"""
    protein_ids = set()
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith("_non_bgc_cds.json"):
                json_path = os.path.join(root, file)
                try:
                    with open(json_path) as f:
                        cds_list = json.load(f)
                        for cds in cds_list:
                            pid = cds.get("protein_id", "")
                            if pid:
                                protein_ids.add(pid)
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")

    protein_ids = list(protein_ids)
    print(
        f"[load_non_bgc_protein_ids] 共加载 {len(protein_ids)} 个非BGC protein_id")
    return protein_ids


def supplement_bgc_cds(bgc_cds_positions, bgc_protein_ids, non_bgc_cds, target_length=128):
    current_length = len(bgc_cds_positions)
    if current_length >= target_length:
        return bgc_cds_positions, bgc_protein_ids, 0, current_length - 1

    n_to_add = target_length - current_length
    # 前补充数量随机，后补充数量为剩下的
    left_add = random.randint(0, n_to_add)
    right_add = n_to_add - left_add

    # 随机选择非BGC CDS
    if len(non_bgc_cds) == 0:
        raise ValueError("non_bgc_cds 为空，无法补充！")

    # 随机打乱 non_bgc_cds
    random.shuffle(non_bgc_cds)

    # 如果不够，则循环取
    if len(non_bgc_cds) < n_to_add:
        non_bgc_cds = non_bgc_cds * (n_to_add // len(non_bgc_cds) + 1)

    # 随机选择前后各一半
    left_cds = non_bgc_cds[:left_add]
    right_cds = non_bgc_cds[left_add:left_add + right_add]

    new_cds_positions = []
    new_protein_ids = []

    # 添加前面
    for cds in left_cds:
        if "protein_id" in cds and cds["protein_id"]:
            new_protein_ids.append(cds["protein_id"])
        elif "locus_tag" in cds and cds["locus_tag"]:
            new_protein_ids.append(cds["locus_tag"])
        elif "gene" in cds and cds["gene"]:
            new_protein_ids.append(cds["gene"])
        else:
            new_protein_ids.append("unknown")
        new_cds_positions.append((cds["start"], cds["end"]))

    left_pad = len(left_cds)
    new_cds_positions.extend(bgc_cds_positions)
    new_protein_ids.extend(bgc_protein_ids)

    # 添加后面
    for cds in right_cds:
        if "protein_id" in cds and cds["protein_id"]:
            new_protein_ids.append(cds["protein_id"])
        elif "locus_tag" in cds and cds["locus_tag"]:
            new_protein_ids.append(cds["locus_tag"])
        elif "gene" in cds and cds["gene"]:
            new_protein_ids.append(cds["gene"])
        else:
            new_protein_ids.append("unknown")
        new_cds_positions.append((cds["start"], cds["end"]))

    right_pad = left_pad + len(bgc_cds_positions) - 1
    # 最终长度校验
    assert len(
        new_cds_positions) == target_length, f"补充后CDS总数{len(new_cds_positions)}不等于目标{target_length}"
    return new_cds_positions, new_protein_ids, left_pad, right_pad


def parse_gbk_dir(gbk_dir, out_json, out_protein_json, non_bgc_dir=None, target_length=128, augment_times=5):
    mapping = {}
    bgc_to_proteins = {}

    # 1. 加载所有非 BGC CDS
    non_bgc_cds_list = []
    if non_bgc_dir:
        non_bgc_cds_list = load_non_bgc_cds(non_bgc_dir)
        print(f"Loaded {len(non_bgc_cds_list)} non-BGC CDS in total")
        # 只用最后 300 万个
        if len(non_bgc_cds_list) > 1000000:
            non_bgc_cds_list = non_bgc_cds_list[-1000000:]
            print(f"仅使用最后 100 万个 non-BGC CDS 进行正样本填充")

    # 以追加模式打开jsonlines文件
    with open(out_json, "a") as f_map, open(out_protein_json, "a") as f_prot:
        for fn in sorted(os.listdir(gbk_dir)):
            if not fn.lower().endswith(".gbk"):
                continue
            bgc_id = os.path.splitext(fn)[0]
            path = os.path.join(gbk_dir, fn)
            original_cds_positions, subregions, original_protein_ids = parse_gbk_text(
                path, bgc_id)

            for i in range(1, augment_times + 1):
                protein_ids = original_protein_ids.copy()
                cds_positions = original_cds_positions.copy()
                new_bgc_id = f"{bgc_id}_aug_{i}"

                if non_bgc_dir and len(protein_ids) < target_length:
                    random.shuffle(non_bgc_cds_list)
                    n_to_add = target_length - len(protein_ids)
                    left_add = random.randint(0, n_to_add)
                    right_add = n_to_add - left_add
                    if len(non_bgc_cds_list) < n_to_add:
                        temp_cds = non_bgc_cds_list * \
                            (n_to_add // len(non_bgc_cds_list) + 1)
                    else:
                        temp_cds = non_bgc_cds_list
                    left_cds = temp_cds[:left_add]
                    right_cds = temp_cds[left_add:left_add + right_add]
                    new_cds_positions = []
                    new_protein_ids = []
                    for cds in left_cds:
                        if "protein_id" in cds and cds["protein_id"]:
                            new_protein_ids.append(cds["protein_id"])
                        elif "locus_tag" in cds and cds["locus_tag"]:
                            new_protein_ids.append(cds["locus_tag"])
                        elif "gene" in cds and cds["gene"]:
                            new_protein_ids.append(cds["gene"])
                        else:
                            new_protein_ids.append("unknown")
                        new_cds_positions.append((cds["start"], cds["end"]))
                    left_pad = len(left_cds)
                    new_cds_positions.extend(cds_positions)
                    new_protein_ids.extend(protein_ids)
                    for cds in right_cds:
                        if "protein_id" in cds and cds["protein_id"]:
                            new_protein_ids.append(cds["protein_id"])
                        elif "locus_tag" in cds and cds["locus_tag"]:
                            new_protein_ids.append(cds["locus_tag"])
                        elif "gene" in cds and cds["gene"]:
                            new_protein_ids.append(cds["gene"])
                        else:
                            new_protein_ids.append("unknown")
                        new_cds_positions.append((cds["start"], cds["end"]))
                    right_pad = left_pad + len(cds_positions) - 1
                    protein_ids = new_protein_ids
                else:
                    left_pad = 0
                    right_pad = len(original_cds_positions) - 1

                if subregions and "type" in subregions[0] and subregions[0]["type"]:
                    main_type = subregions[0]["type"]
                else:
                    main_type = map_to_7class(bgc_id)

                regions = [{
                    "start": left_pad,
                    "end": right_pad,
                    "type": main_type
                }]

                print(f"写入 mapping: {new_bgc_id}")
                f_map.write(json.dumps(
                    {new_bgc_id: regions}, ensure_ascii=False) + "\n")
                f_map.flush()
                f_prot.write(json.dumps(
                    {new_bgc_id: protein_ids}, ensure_ascii=False) + "\n")
                f_prot.flush()

                print(
                    f"{new_bgc_id}: 前补={left_pad}, BGC区间={left_pad}-{right_pad}, BGC长度={len(original_cds_positions)}, 总数={target_length}")

    print(f"done: 已追加写入 {out_json}, {out_protein_json}")
    print(f"每个 BGC 仅保留扩增样本，扩增后总数: 实时可查")


def add_real_negative_samples(mapping_json, proteins_json, output_dir, target_length=128, min_negatives=20000):
    import json
    from tqdm import tqdm

    # 统计已有负样本数量
    neg_count = 0
    if os.path.exists(mapping_json):
        with open(mapping_json) as f:
            for line in f:
                if line.startswith('{"NEG_'):
                    neg_count += 1

    # 以追加模式打开jsonlines文件
    with open(mapping_json, "a") as f_map, open(proteins_json, "a") as f_prot:
        all_json_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith("_non_bgc_cds.json"):
                    all_json_files.append(os.path.join(root, file))

        pbar = tqdm(total=min_negatives, desc="负样本生成/写入数")
        written = 0
        for json_path in all_json_files:
            try:
                with open(json_path) as f:
                    cds_list = json.load(f)

                file_written = 0  # 统计当前文件生成的负样本数
                # 按顺序每128个CDS构造一个负样本，每次向后跳30个CDS
                i = 0
                while i + target_length <= len(cds_list):
                    group = cds_list[i:i + target_length]

                    # 检查这组CDS是否都有protein_id
                    has_all_protein_id = True
                    for cds in group:
                        if not ("protein_id" in cds and cds["protein_id"]):
                            has_all_protein_id = False
                            break

                    if not has_all_protein_id:
                        i += 1  # 如果没有protein_id，向后移动一个位置
                        continue

                    neg_id = f"NEG_{neg_count:05d}"
                    # 写入蛋白质ID
                    f_prot.write(json.dumps({neg_id: [
                        cds["protein_id"] for cds in group
                    ]}, ensure_ascii=False) + "\n")
                    f_prot.flush()  # 实时写入

                    # 写入mapping，CDS索引固定为0-127
                    f_map.write(json.dumps({neg_id: [{
                        "start": 0,
                        "end": target_length - 1,
                        "type": "no-object"
                    }]}, ensure_ascii=False) + "\n")
                    f_map.flush()  # 实时写入

                    neg_count += 1
                    written += 1
                    file_written += 1
                    pbar.update(1)
                    if written >= min_negatives:
                        break
                    i += 30  # 每次向后跳30个CDS

                print(
                    f"文件 {os.path.basename(json_path)} 生成了 {file_written} 个负样本")
                if written >= min_negatives:
                    break
            except Exception as e:
                print(f"处理文件 {json_path} 时出错: {e}")
                continue

        pbar.close()
    print(f"已追加 {written} 个负样本到 {mapping_json} 和 {proteins_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gbk_dir", help="GBK 输入目录")
    p.add_argument("--out_json", help="输出 JSON 路径")
    p.add_argument("--out_protein_json", help="输出蛋白顺序 JSON 路径")
    p.add_argument("--non_bgc_dir", help="非 BGC CDS JSON 文件目录")
    p.add_argument("--target_length", type=int, default=128, help="目标 CDS 数量")
    p.add_argument("--augment_times", type=int, default=5, help="每个 BGC 扩充的次数")
    p.add_argument("--add_negatives", action="store_true", help="是否添加负样本")
    p.add_argument("--mapping_json", type=str,
                   default="data/bgc_mapping.json", help="mapping json 路径")
    p.add_argument("--proteins_json", type=str,
                   default="data/bgc_proteins.json", help="proteins json 路径")
    p.add_argument("--output_dir", type=str,
                   default="data/output", help="output 目录")
    p.add_argument("--min_negatives", type=int, default=20000, help="最少负样本数量")
    args = p.parse_args()

    if not args.add_negatives:
        # 只有做正样本时才检查这三个参数
        if not args.gbk_dir or not args.out_json or not args.out_protein_json:
            p.error(
                "--gbk_dir, --out_json, --out_protein_json are required unless --add_negatives is set")
        parse_gbk_dir(args.gbk_dir, args.out_json, args.out_protein_json,
                      args.non_bgc_dir, args.target_length, args.augment_times)
    if args.add_negatives:
        add_real_negative_samples(args.mapping_json, args.proteins_json, args.output_dir,
                                  args.target_length, args.min_negatives)
# python scripts/parse_gbk.py --gbk_dir data/val_gbk --out_json data/val/bgc_mapping.json --out_protein_json data/val/bgc_proteins.json --non_bgc_dir data/output --target_length 128 --augment_times 1

# python scripts/parse_gbk.py --add_negatives --mapping_json data/bgc_mapping.json --proteins_json data/bgc_proteins.json --output_dir data/output --target_length 128 --min_negatives 20000
