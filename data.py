# type: ignore
"""
数据集模块
提供用于BGC检测的数据集类，包括固定长度数据集和平衡数据集

该模块包含两个主要的数据集类：
1. FixedClusterDataset: 固定长度的簇级数据集
2. BalancedClusterDataset: 平衡的簇级数据集，支持类别平衡策略
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
from collections import Counter
from sklearn.utils import resample


class FixedClusterDataset(Dataset):
    """
    固定长度的簇级数据集
    每个样本就是一个 BGC 的蛋白质级嵌入序列，截断/补齐到固定长度
    """

    def __init__(self,
                 mapping,
                 emb_dir: str,
                 max_tokens: int = 128,
                 seq_id_list: list = None,
                 binary_mode: bool = False):
        if isinstance(mapping, str):
            with open(mapping) as f:
                full = json.load(f)
        elif isinstance(mapping, dict):
            full = mapping
        else:
            raise ValueError("mapping 必须是 JSON 文件路径(str)或已 load 的 dict")

        if seq_id_list is not None:
            self.mapping = {k: full[k] for k in seq_id_list if k in full}
        else:
            self.mapping = full

        self.emb_dir = emb_dir
        self.max_tokens = max_tokens
        self.binary_mode = binary_mode
        self.bgc_ids = sorted(self.mapping.keys())

        labels = {self._canonical_label(r["type"]) for regs in self.mapping.values() for r in regs}
        if self.binary_mode:
            sorted_labels = ['bgc', 'no-object']
        else:
            sorted_labels = sorted([l for l in labels if l != 'no-object']) + \
                (['no-object'] if 'no-object' in labels else [])
        self.label2id = {l: i for i, l in enumerate(sorted_labels)}
        self.num_classes = len(self.label2id)

    def _canonical_label(self, label: str) -> str:
        if self.binary_mode and label != 'no-object':
            return 'bgc'
        return label

    def __len__(self):
        return len(self.bgc_ids)

    def __getitem__(self, idx):
        bgc = self.bgc_ids[idx]
        emb = np.load(os.path.join(self.emb_dir, f"{bgc}.npy"))
        emb = torch.from_numpy(emb).float()
        L, C = emb.shape
        if C != 256:
            raise ValueError(f"嵌入维度错误：期望 256，得到 {C}")

        if L >= self.max_tokens:
            emb = emb[:self.max_tokens]
            valid = self.max_tokens
        else:
            pad = self.max_tokens - L
            emb = F.pad(emb, (0, 0, 0, pad))
            valid = L

        mask = torch.zeros(self.max_tokens, dtype=torch.bool)
        mask[valid:] = True

        boxes, labels = [], []
        for r in self.mapping[bgc]:
            label = self._canonical_label(r.get("type", 'no-object'))
            if label == 'no-object':
                continue
            s, e = r["start"], r["end"]
            if e > 0 and s < self.max_tokens:
                a = max(0, s)
                b = min(e, self.max_tokens)
                boxes.append([a / self.max_tokens, b / self.max_tokens])
                labels.append(self.label2id[label])

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 2), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return emb, mask, {
            "boxes": boxes,
            "labels": labels,
            "orig_size": torch.tensor([valid, 1]),
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 1] - boxes[:, 0]).clamp(min=0),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "size": torch.tensor([valid, 1])
        }


class BalancedClusterDataset(Dataset):
    """
    平衡的簇级数据集
    通过上采样和下采样来平衡类别分布，解决类别不平衡问题
    """

    def __init__(self,
                 mapping,
                 emb_dir: str,
                 max_tokens: int = 128,
                 seq_id_list: list = None,
                 balance_strategy: str = 'upsample',
                 target_samples: int = None,
                 binary_mode: bool = False):
        if isinstance(mapping, str):
            with open(mapping) as f:
                full = json.load(f)
        elif isinstance(mapping, dict):
            full = mapping
        else:
            raise ValueError("mapping 必须是 JSON 文件路径(str)或已 load 的 dict")

        if seq_id_list is not None:
            self.mapping = {k: full[k] for k in seq_id_list if k in full}
        else:
            self.mapping = full

        self.emb_dir = emb_dir
        self.max_tokens = max_tokens
        self.balance_strategy = balance_strategy
        self.target_samples = target_samples
        self.binary_mode = binary_mode

        self.label2id = {}
        self.id2label = {}
        self.bgc_by_type = {}

        all_types = set()
        for bgc_id, regions in self.mapping.items():
            for region in regions:
                all_types.add(self._canonical_label(region['type']))

        if self.binary_mode:
            sorted_types = ['bgc', 'no-object']
        else:
            sorted_types = sorted([t for t in all_types if t != 'no-object'])
            if 'no-object' in all_types:
                sorted_types.append('no-object')

        for i, reg_type in enumerate(sorted_types):
            self.label2id[reg_type] = i
            self.id2label[i] = reg_type
            self.bgc_by_type[reg_type] = []

        for bgc_id in sorted(self.mapping.keys()):
            seen_types = set()
            regions = self.mapping[bgc_id]
            for region in regions:
                reg_type = self._canonical_label(region['type'])
                self.bgc_by_type[reg_type].append(bgc_id)
                seen_types.add(reg_type)

        self.num_classes = len(self.label2id)
        self.bgc_ids = self._balance_dataset()
        self.class_weights = self._compute_class_weights()

    def _canonical_label(self, label: str) -> str:
        if self.binary_mode and label != 'no-object':
            return 'bgc'
        return label

    def _balance_dataset(self):
        if self.balance_strategy == 'upsample':
            return self._upsample_dataset()
        elif self.balance_strategy == 'downsample':
            return self._downsample_dataset()
        else:
            return list(self.mapping.keys())

    def _upsample_dataset(self):
        balanced_bgc_ids = []
        max_samples = max(len(bgc_list)
                          for bgc_list in self.bgc_by_type.values())
        for reg_type, bgc_list in self.bgc_by_type.items():
            if len(bgc_list) < max_samples:
                upsampled = resample(bgc_list,
                                     replace=True,
                                     n_samples=max_samples,
                                     random_state=42)
                balanced_bgc_ids.extend(upsampled)
            else:
                balanced_bgc_ids.extend(bgc_list)
        return balanced_bgc_ids

    def _downsample_dataset(self):
        balanced_bgc_ids = []
        min_samples = min(len(bgc_list)
                          for bgc_list in self.bgc_by_type.values())
        for reg_type, bgc_list in self.bgc_by_type.items():
            if len(bgc_list) > min_samples:
                downsampled = resample(bgc_list,
                                       replace=False,
                                       n_samples=min_samples,
                                       random_state=42)
                balanced_bgc_ids.extend(downsampled)
            else:
                balanced_bgc_ids.extend(bgc_list)
        return balanced_bgc_ids

    def _compute_class_weights(self):
        class_counts = Counter()
        for bgc_id in self.bgc_ids:
            for region in self.mapping[bgc_id]:
                class_counts[self._canonical_label(region['type'])] += 1
        total = sum(class_counts.values())
        weights = {cls: total / (len(class_counts) * count)
                   for cls, count in class_counts.items()}
        return weights

    def get_sampler(self):
        if self.balance_strategy == 'weighted':
            weights = []
            for bgc_id in self.bgc_ids:
                seen = set()
                for region in self.mapping[bgc_id]:
                    label = self._canonical_label(region['type'])
                    if label in seen:
                        continue
                    seen.add(label)
                    weights.append(self.class_weights[label])
            return WeightedRandomSampler(weights, len(weights))
        return None

    def __len__(self):
        return len(self.bgc_ids)

    def __getitem__(self, idx):
        bgc_id = self.bgc_ids[idx]
        regions = self.mapping[bgc_id]
        emb_path = os.path.join(self.emb_dir, f"{bgc_id}.npy")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"找不到嵌入文件: {emb_path}")
        emb = torch.tensor(np.load(emb_path), dtype=torch.float32)
        n_prots = emb.shape[0]

        if n_prots > self.max_tokens:
            emb = emb[:self.max_tokens]
            mask = torch.zeros(self.max_tokens, dtype=torch.bool)
            valid = self.max_tokens
        else:
            pad = self.max_tokens - n_prots
            emb = F.pad(emb, (0, 0, 0, pad))
            mask = torch.cat([
                torch.zeros(n_prots, dtype=torch.bool),
                torch.ones(pad, dtype=torch.bool)
            ])
            valid = n_prots

        boxes, labels = [], []
        for region in regions:
            label = self._canonical_label(region['type'])
            if label == 'no-object':
                continue
            start = region['start']
            end = region['end']
            if end > 0 and start < self.max_tokens:
                a = max(0, start)
                b = min(end, self.max_tokens)
                boxes.append([a / self.max_tokens, b / self.max_tokens])
                labels.append(self.label2id[label])

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 2), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return emb, mask, {
            "boxes": boxes,
            "labels": labels,
            "orig_size": torch.tensor([valid, 1]),
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 1] - boxes[:, 0]).clamp(min=0),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "size": torch.tensor([valid, 1])
        }
