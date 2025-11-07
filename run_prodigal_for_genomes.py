#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量运行 Prodigal：将原始核酸 FASTA -> 蛋白质 FASTA（genome.protein_*.fa）

目标：为 process_prediction_data_fixed.py 生成标准输入文件：
  GeneOutputs/proteins/genome.protein_<genome_id>.fa

要点：
- 可选地将输入 FASTA 中的 contig 名重命名为 <genome_id>_<idx>，
  以便 Prodigal 产生的蛋白序列 ID 形如 <genome_id>_<contigIdx>_<cdsIdx>。
- 需要系统已安装 prodigal 可执行文件（PATH 可访问）。
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import site
import sysconfig
from typing import List


def which(exe: str) -> str:
    path = shutil.which(exe)
    return path or ""


def _append_user_bin_to_path() -> None:
    """将用户级 bin 目录追加到 PATH，便于刚通过 pip --user 安装的可执行文件被找到。"""
    user_base = getattr(site, "USER_BASE", None) or sysconfig.get_config_var("userbase")
    if user_base:
        user_bin = os.path.join(user_base, "bin")
        if os.path.isdir(user_bin) and user_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = user_bin + os.pathsep + os.environ.get("PATH", "")


def ensure_prodigal_available(tsinghua_mirror: bool = True) -> str:
    """确保 prodigal 可用；若未安装，尝试使用清华源通过 pip 安装。

    返回 prodigal 可执行文件的绝对路径；若仍不可用则抛出异常并给出指引。
    """
    path = which("prodigal")
    if path:
        return path

    # 优先尝试使用清华镜像安装到用户目录
    mirror = "https://pypi.tuna.tsinghua.edu.cn/simple" if tsinghua_mirror else None
    install_cmd = [sys.executable, "-m", "pip", "install", "prodigal"]
    # 使用 --user 提高在集群环境下的成功率
    install_cmd.insert(3, "--user")
    if mirror:
        install_cmd += ["-i", mirror]
    try:
        print("[info] 未检测到 prodigal，正在通过 pip 安装（--user）...")
        subprocess.check_call(install_cmd)
    except Exception as e:
        print(f"[warn] pip --user 安装 prodigal 失败：{e}")

    # 刷新 PATH 并重查
    _append_user_bin_to_path()
    path = which("prodigal")
    if path:
        return path

    # 再尝试不带 --user 的安装（某些环境已在虚拟环境/容器中）
    install_cmd_no_user = [sys.executable, "-m", "pip", "install", "prodigal"]
    if mirror:
        install_cmd_no_user += ["-i", mirror]
    try:
        print("[info] 正在尝试 pip 安装（无 --user）...")
        subprocess.check_call(install_cmd_no_user)
    except Exception as e:
        print(f"[warn] pip 安装 prodigal（无 --user）失败：{e}")

    path = which("prodigal")
    if path:
        return path

    raise FileNotFoundError(
        "未能自动安装或找到 'prodigal' 可执行文件。请尝试以下方式之一：\n"
        "  1) conda install -c bioconda prodigal\n"
        "  2) apt-get install prodigal（如有sudo权限）\n"
        "  3) 手动将 prodigal 放入 PATH 后重试"
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_genome_id(path: str) -> str:
    """从文件名推断 genome_id（去扩展名）。"""
    base = os.path.basename(path)
    # 去除常见扩展名
    for suf in (".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz", ".fna.gz"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return base


def iter_input_fastas(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        files = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if f.endswith((".fa", ".fna", ".fasta"))
        ]
        return files
    return [input_path]


def rewrite_contig_headers(src_fa: str, dst_fa: str, genome_id: str) -> None:
    """将 FASTA 的序列头重命名为 >{genome_id}_{i}（i 从 1 开始）。"""
    idx = 0
    with open(src_fa, "r") as fin, open(dst_fa, "w") as fout:
        for line in fin:
            if line.startswith(">"):
                idx += 1
                fout.write(f">{genome_id}_{idx}\n")
            else:
                fout.write(line)


def run_prodigal(nucl_fa: str, out_prot_fa: str, mode: str) -> None:
    prodigal = ensure_prodigal_available(tsinghua_mirror=True)
    # -q 静默，-p meta 用于多样 contig / 宏基因组场景；普通基因组可用 -p single
    cmd = [
        prodigal,
        "-i", nucl_fa,
        "-a", out_prot_fa,
        "-q",
        "-p", mode,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Prodigal 运行失败: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def process_one(input_fa: str, output_dir: str, mode: str, rename_contigs: bool) -> str:
    genome_id = normalize_genome_id(input_fa)
    ensure_dir(output_dir)

    with tempfile.TemporaryDirectory() as td:
        nucl_for_prodigal = input_fa
        if rename_contigs:
            nucl_for_prodigal = os.path.join(td, "renamed.fa")
            rewrite_contig_headers(input_fa, nucl_for_prodigal, genome_id)

        out_tmp = os.path.join(td, "proteins.fa")
        run_prodigal(nucl_for_prodigal, out_tmp, mode)

        final_path = os.path.join(output_dir, f"genome.protein_{genome_id}.fa")
        shutil.move(out_tmp, final_path)
        return final_path


def main() -> None:
    ap = argparse.ArgumentParser(description="批量运行 Prodigal 生成 genome.protein_*.fa")
    ap.add_argument("--input", required=True, help="输入：单个FASTA文件或目录（*.fa|*.fna|*.fasta）")
    ap.add_argument("--output_dir", default="/share/org/BGI/bgi_suncl/project/GeneOutputs/proteins",
                    help="输出目录（默认写入 GeneOutputs/proteins）")
    ap.add_argument("--mode", choices=["meta", "single"], default="meta",
                    help="Prodigal 预测模式：单基因组用 single，宏基因组/多contig 用 meta（默认）")
    ap.add_argument("--no_rename_contigs", action="store_true",
                    help="不重命名 contig（默认会重命名为 <genome_id>_<idx>）")
    args = ap.parse_args()

    inputs = iter_input_fastas(args.input)
    if not inputs:
        print("未找到输入FASTA", file=sys.stderr)
        sys.exit(2)

    rename = not args.no_rename_contigs
    produced = []
    for fa in inputs:
        try:
            outp = process_one(fa, args.output_dir, args.mode, rename)
            produced.append(outp)
            print(f"OK: {fa} -> {outp}")
        except Exception as e:
            print(f"FAILED: {fa}: {e}", file=sys.stderr)
            sys.exit(1)

    print("\n生成完成：")
    for p in produced:
        print(f"  - {p}")


if __name__ == "__main__":
    main()


