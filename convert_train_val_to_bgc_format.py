import json
import os
import argparse
from typing import List, Dict, Any, Tuple


def parse_labels_to_regions(labels: List[int], bgc_type: str) -> List[Dict[str, Any]]:
    regions = []
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == 1:
            start = i
            j = i + 1
            while j < n and labels[j] == 1:
                j += 1
            end = j - 1
            regions.append({
                "start": start,
                "end": end,
                "type": bgc_type if bgc_type else "no-object"
            })
            i = j
        else:
            i += 1
    return regions


def load_large_json(path: str) -> Any:
    with open(path, 'r') as f:
        return json.load(f)


def normalize_labels_field(raw) -> List[int]:
    if isinstance(raw, list):
        return [int(x) for x in raw]
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                arr = json.loads(s)
                return [int(x) for x in arr]
            except Exception:
                pass
        # fallback: split by comma
        s = s.strip('[]')
        parts = [p.strip() for p in s.split(',') if p.strip()]
        return [int(p) for p in parts]
    raise ValueError("Unsupported bgc_labels format")


def infer_keys(sample: Dict[str, Any]) -> Tuple[str, str, str]:
    # returns (cds_list_key, labels_key, class_key)
    labels_key = None
    class_key = None
    cds_key = None
    for k in sample.keys():
        lk = k.lower()
        if lk == 'bgc_labels':
            labels_key = k
        elif lk == 'bgc_class':
            class_key = k
    # heuristics: remaining list of strings of length >= 1 assumed to be CDS ids
    for k, v in sample.items():
        if k in (labels_key, class_key):
            continue
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            cds_key = k
            break
    if not (labels_key and class_key and cds_key):
        raise KeyError(
            f"Cannot infer keys from sample keys={list(sample.keys())}")
    return cds_key, labels_key, class_key


def build_outputs(items: List[Dict[str, Any]], prefix: str) -> Tuple[Dict[str, Any], Dict[str, List[str]], Dict[str, str]]:
    mapping: Dict[str, Any] = {}
    proteins: Dict[str, List[str]] = {}
    sequences: Dict[str, str] = {}

    # infer keys from first item
    cds_key, labels_key, class_key = infer_keys(items[0])

    for idx, sample in enumerate(items):
        sample_key = f"{prefix}_{idx:08d}"
        cds_ids = sample[cds_key]
        labels = normalize_labels_field(sample[labels_key])
        classes = sample.get(class_key) or []
        bgc_type = classes[0] if (isinstance(classes, list) and len(
            classes) > 0 and classes[0]) else "no-object"

        # regions
        regions = parse_labels_to_regions(labels, bgc_type)
        mapping[sample_key] = regions

        # proteins list
        proteins[sample_key] = cds_ids

        # sequences: expect sample may contain sequences mapping, try common fields
        # 1) dict: {cds_id: seq} (e.g., 'orf_sequences')
        # 2) list same order as cds_ids: sample.get('sequences')
        # 3) list of objects: {id, sequence}
        seq_map: Dict[str, str] = {}
        if isinstance(sample.get('cds_sequences'), dict):
            seq_map = sample['cds_sequences']
        elif isinstance(sample.get('orf_sequences'), dict):
            # antiSMASH风格：orf_sequences: {protein_id: aa_sequence}
            seq_map = sample['orf_sequences']
        elif isinstance(sample.get('sequences'), list) and len(sample['sequences']) == len(cds_ids):
            for _id, _seq in zip(cds_ids, sample['sequences']):
                if isinstance(_seq, str):
                    seq_map[_id] = _seq
        elif isinstance(sample.get('cds_seq'), list) and len(sample['cds_seq']) == len(cds_ids):
            for _id, _seq in zip(cds_ids, sample['cds_seq']):
                if isinstance(_seq, str):
                    seq_map[_id] = _seq
        elif isinstance(sample.get('cds'), list):
            for obj in sample['cds']:
                if isinstance(obj, dict):
                    _id = obj.get('id') or obj.get('name') or obj.get('cds_id')
                    _seq = obj.get('sequence') or obj.get('seq')
                    if isinstance(_id, str) and isinstance(_seq, str):
                        seq_map[_id] = _seq

        for _id in cds_ids:
            if _id in seq_map and isinstance(seq_map[_id], str):
                sequences[_id] = seq_map[_id]

    return mapping, proteins, sequences


def convert(train_path: str, val_path: str, out_root: str):
    os.makedirs(out_root, exist_ok=True)
    out_train = os.path.join(out_root, 'train')
    out_val = os.path.join(out_root, 'val')
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_val, exist_ok=True)

    # load inputs (array of objects expected)
    train_items = load_large_json(train_path)
    val_items = load_large_json(val_path)
    if not isinstance(train_items, list) or not isinstance(val_items, list):
        raise ValueError("Expected input JSON files to be arrays of objects")

    train_mapping, train_proteins, train_seqs = build_outputs(
        train_items, prefix='TRAIN')
    val_mapping, val_proteins, val_seqs = build_outputs(
        val_items, prefix='VAL')

    # merge sequences to reduce duplicates
    merged_seqs = dict(train_seqs)
    merged_seqs.update(val_seqs)

    # dump
    with open(os.path.join(out_train, 'bgc_mapping.json'), 'w') as f:
        json.dump(train_mapping, f)
    with open(os.path.join(out_train, 'bgc_proteins.json'), 'w') as f:
        json.dump(train_proteins, f)
    with open(os.path.join(out_train, 'cds_sequences.json'), 'w') as f:
        json.dump(merged_seqs, f)

    with open(os.path.join(out_val, 'bgc_mapping.json'), 'w') as f:
        json.dump(val_mapping, f)
    with open(os.path.join(out_val, 'bgc_proteins.json'), 'w') as f:
        json.dump(val_proteins, f)
    with open(os.path.join(out_val, 'cds_sequences.json'), 'w') as f:
        json.dump(merged_seqs, f)


def main():
    p = argparse.ArgumentParser(
        description='Convert train/val JSON to bgc_mapping/proteins/cds_sequences format')
    p.add_argument('--train_json', default='trainnew.json')
    p.add_argument('--val_json', default='validationnew.json')
    p.add_argument('--out_dir', default='data/converted')
    args = p.parse_args()

    convert(args.train_json, args.val_json, args.out_dir)


if __name__ == '__main__':
    main()
