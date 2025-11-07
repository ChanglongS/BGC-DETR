#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from Bio import SeqIO
from Bio.SeqFeature import FeatureLocation
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_bgc_regions(antismash_dir):
    """从 antiSMASH 输出目录提取 BGC 区域

    Args:
        antismash_dir: antiSMASH 输出目录
    Returns:
        list: BGC 区域列表，每个元素为 (start, end)
    """
    bgc_regions = []
    try:
        # 查找所有 region 文件
        region_files = list(Path(antismash_dir).glob("*.region*.gbk"))
        logger.info(
            f"Found {len(region_files)} region files in {antismash_dir}")

        for region_file in region_files:
            try:
                record = SeqIO.read(region_file, "genbank")
                for feature in record.features:
                    if feature.type == "region":
                        start = feature.location.start
                        end = feature.location.end
                        bgc_regions.append((start, end))
                        logger.debug(f"Found BGC region: {start}-{end}")
            except Exception as e:
                logger.error(
                    f"Error reading region file {region_file}: {str(e)}")
    except Exception as e:
        logger.error(
            f"Error processing antiSMASH directory {antismash_dir}: {str(e)}")

    return bgc_regions


def extract_non_bgc_cds(gbk_file, bgc_regions):
    """提取非 BGC 区域的 CDS

    Args:
        gbk_file: GenBank 文件路径
        bgc_regions: BGC 区域列表，每个元素为 (start, end)
    Returns:
        list: 非 BGC 区域的 CDS 列表
    """
    non_bgc_cds = []
    try:
        # 使用 SeqIO.parse 而不是 SeqIO.read 来处理多记录文件
        for record in SeqIO.parse(gbk_file, "genbank"):
            for feature in record.features:
                if feature.type == "CDS":
                    # 检查 CDS 是否在 BGC 区域内
                    is_in_bgc = False
                    for start, end in bgc_regions:
                        if start <= feature.location.start and feature.location.end <= end:
                            is_in_bgc = True
                            break

                    if not is_in_bgc:
                        # 提取 CDS 信息
                        cds_info = {
                            "start": feature.location.start,
                            "end": feature.location.end,
                            "strand": feature.location.strand,
                            "locus_tag": feature.qualifiers.get("locus_tag", [""])[0],
                            "product": feature.qualifiers.get("product", [""])[0],
                            "protein_id": feature.qualifiers.get("protein_id", [""])[0],
                            "translation": feature.qualifiers.get("translation", [""])[0]
                        }
                        non_bgc_cds.append(cds_info)
                        logger.debug(
                            f"Found non-BGC CDS: {cds_info['locus_tag']}")
    except Exception as e:
        logger.error(f"Error extracting non-BGC CDS from {gbk_file}: {str(e)}")

    return non_bgc_cds


def process_genome(gbk_file, output_dir):
    """处理单个基因组文件

    Args:
        gbk_file: GenBank 文件路径
        output_dir: 输出目录
    Returns:
        bool: 处理是否成功
    """
    try:
        gbk_file = Path(gbk_file)
        genome_id = gbk_file.stem
        genome_output_dir = Path(output_dir) / genome_id
        genome_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = genome_output_dir / f"{genome_id}_non_bgc_cds.json"

        # ====== 新增：如果结果已存在，直接跳过 ======
        if output_file.exists():
            logger.info(f"{output_file} 已存在，跳过该基因组")
            return True
        # =========================================

        # 运行 antiSMASH
        logger.info(f"Processing {genome_id} with antiSMASH")
        antismash_cmd = f"antismash {gbk_file} --output-dir {genome_output_dir} --minimal"
        os.system(antismash_cmd)

        # 提取 BGC 区域
        bgc_regions = extract_bgc_regions(genome_output_dir)
        logger.info(f"Found {len(bgc_regions)} BGC regions in {genome_id}")

        # 提取非 BGC 区域的 CDS
        non_bgc_cds = extract_non_bgc_cds(gbk_file, bgc_regions)
        logger.info(f"Found {len(non_bgc_cds)} non-BGC CDS in {genome_id}")
        # logger.info(f"non_bgc_cds: {non_bgc_cds}")
        # 保存结果
        with open(output_file, "w") as f:
            json.dump(non_bgc_cds, f, indent=2)
        logger.info(f"Saved results to {output_file}")

        return True
    except Exception as e:
        logger.error(f"Error processing {gbk_file}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract non-BGC CDS from genome files using antiSMASH")
    parser.add_argument("--genome_dir", required=True,
                        help="Directory containing genome files")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理所有基因组文件
    gbk_files = list(Path(args.genome_dir).glob("*.gbk")) + \
        list(Path(args.genome_dir).glob("*.gbff"))
    logger.info(f"Found {len(gbk_files)} genome files to process")

    # 使用进度条显示处理进度
    for gbk_file in tqdm(gbk_files, desc="Processing genomes"):
        process_genome(gbk_file, output_dir)


if __name__ == "__main__":
    main()


# # 运行处理脚本
# python process_genomes.py --genome_dir data/nine_genomes_gbff/ --output_dir data/output/
