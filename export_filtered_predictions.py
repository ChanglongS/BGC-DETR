#type: ignore
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export filtered BGC-DETR predictions with confidence threshold.

This utility takes a `detailed_predictions_*.json` file together with the
`bgc_proteins.json` mapping, filters prediction boxes whose confidence exceeds
the given threshold, and produces CSV summaries describing which CDS are
considered BGC-like – similar to the `bgc_prophet_output/proteins*.csv`
structure.

Outputs (written to the specified directory):

1. `segments_filtered.csv`
   Each row corresponds to a segment-level prediction (a retained DETR box) and
   lists the covered CDS identifiers alongside class / confidence information.

2. `cds_filtered.csv`
   Each row corresponds to an individual CDS with the highest-confidence class
   predicted for it.

3. `segments_like_prophet.csv`
   A summary in the style of bgc-prophet where the `TDsentence` column contains
   space-separated CDS IDs predicted as BGC for each segment and `isBGC` is set
   to "Yes" when there exists at least one retained prediction.

Example usage:

    python scripts/export_filtered_predictions.py \
        --detailed_json GeneOutputs/my_predictions/prediction_all_models/\\
            detailed_predictions_checkpoint_fold_1_epoch_89_best_in_interval.json \
        --bgc_proteins BGC-DETR/data/prediction/bgc_proteins.json \
        --output_dir GeneOutputs/filtered_predictions --confidence_threshold 0.5

"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter DETR predictions by confidence")
    parser.add_argument("--detailed_json", required=True,
                        help="Path to detailed_predictions_*.json file")
    parser.add_argument("--bgc_proteins", required=True,
                        help="Path to bgc_proteins.json created for prediction")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Minimum confidence to keep a prediction (default: 0.5)")
    parser.add_argument("--output_dir", default="filtered_predictions",
                        help="Directory to store the exported CSV files")
    return parser.parse_args()


def load_segment_mapping(bgc_proteins_path: str) -> Tuple[List[str], Dict[int, str], Dict[str, List[str]]]:
    with open(bgc_proteins_path, "r", encoding="utf-8") as f:
        seg_to_proteins = json.load(f)

    # The detailed_predictions JSON stores sample_id which corresponds to the
    # index of the segment when iterating in sorted order.
    segment_ids = sorted(seg_to_proteins.keys())
    sample_id_to_segment = {idx: seg_id for idx, seg_id in enumerate(segment_ids)}
    return segment_ids, sample_id_to_segment, seg_to_proteins


def normalize_segment_id(segment_id: str) -> Tuple[str, str]:
    """Extract genome identifier and segment index from PRED segment id."""
    parts = segment_id.split("_")
    if len(parts) < 4 or parts[0] != "PRED":
        return "", ""
    genome = parts[1]
    seg_idx = parts[-1]
    return genome, seg_idx


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    segment_ids, sample_id_to_segment, seg_to_proteins = load_segment_mapping(args.bgc_proteins)

    with open(args.detailed_json, "r", encoding="utf-8") as f:
        detail = json.load(f)

    # Collect outputs
    segment_rows: List[List[str]] = []
    cds_best: Dict[Tuple[str, str], Tuple[str, float, int]] = {}
    prophet_like_rows: List[List[str]] = []

    for sample in detail.get("samples", []):
        sample_id = sample.get("sample_id")
        seg_id = sample_id_to_segment.get(sample_id)
        if seg_id is None:
            continue

        genome, seg_idx = normalize_segment_id(seg_id)
        cds_list = seg_to_proteins[seg_id]

        retained_cds_for_segment: List[str] = []

        for pred in sample.get("predictions", []):
            confidence = float(pred.get("confidence", 0.0))
            if confidence < args.confidence_threshold:
                continue
            class_id = pred.get("class_id")
            class_name = pred.get("class_name") or f"class_{class_id}"
            normalized_class_name = class_name.strip().lower()
            if normalized_class_name in {"ribosomal", "ripp"}:
                class_name = "RiPP"
            elif normalized_class_name in {"pk", "pks"}:
                class_name = "PKS"
            elif normalized_class_name == "nrps":
                class_name = "NRPS"
            elif normalized_class_name == "saccharide":
                class_name = "saccharide"
            elif normalized_class_name == "terpene":
                class_name = "terpene"
            elif normalized_class_name == "other":
                class_name = "other"
            else:
                # Fallback to original (stripped) form
                class_name = class_name.strip()
            if class_id == 6 or class_name.lower() in {"no-object", "background"}:
                continue

            start = max(0, int(pred.get("start_cds", 0)))
            end = min(len(cds_list), int(pred.get("end_cds", start)))
            if end <= start:
                continue

            covered_cds = cds_list[start:end]
            retained_cds_for_segment.extend(covered_cds)

            segment_rows.append([
                seg_id,
                genome,
                seg_idx,
                str(pred.get("query_id")),
                class_name,
                f"{confidence:.6f}",
                str(start),
                str(end),
                " ".join(covered_cds)
            ])

            # Record per-CDS best prediction
            for local_idx, cds_id in enumerate(covered_cds, start=start):
                key = (seg_id, cds_id)
                prev = cds_best.get(key)
                if prev is None or confidence > prev[1]:
                    cds_best[key] = (class_name, confidence, pred.get("query_id"))

        # Build bgc-prophet like row for this segment
        unique_cds = sorted(set(retained_cds_for_segment))
        is_bgc = "Yes" if unique_cds else "No"
        prophet_like_rows.append([
            seg_id,
            " ".join(unique_cds),
            is_bgc,
            " ".join(sorted({normalize_segment_id(seg_id)[0]})) if unique_cds else "",
            str(len(unique_cds))
        ])

    # Write segments_filtered.csv
    segments_csv_path = os.path.join(args.output_dir, "segments_filtered.csv")
    with open(segments_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "segment_id", "genome", "segment_index", "query_id",
            "class_name", "confidence", "start_cds", "end_cds", "cds_ids"
        ])
        writer.writerows(segment_rows)

    # Write cds_filtered.csv (best prediction per CDS)
    cds_csv_path = os.path.join(args.output_dir, "cds_filtered.csv")
    with open(cds_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["segment_id", "cds_id", "class_name", "confidence", "query_id"])
        for (seg_id, cds_id), (class_name, confidence, query_id) in sorted(cds_best.items()):
            writer.writerow([
                seg_id,
                cds_id,
                class_name,
                f"{confidence:.6f}",
                query_id
            ])

    # Write segments_like_prophet.csv for quick inspection
    prophet_like_path = os.path.join(args.output_dir, "segments_like_prophet.csv")
    with open(prophet_like_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "TDsentence", "isBGC", "genome", "cds_count"])
        writer.writerows(prophet_like_rows)

    print(f"Filtered predictions written to {args.output_dir}")
    print(f"  - {segments_csv_path}")
    print(f"  - {cds_csv_path}")
    print(f"  - {prophet_like_path}")


if __name__ == "__main__":
    main()

