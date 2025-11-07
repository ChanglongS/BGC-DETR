# type: ignore
# !/usr/bin/env python3
import math
import sys
import time
from typing import Iterable, Optional, List
from torch import Tensor

import torch
import torch.nn.functional as F
import util.misc as utils
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from collections import Counter, defaultdict, deque
import datetime
import os
import subprocess
import torchvision

# ===== 统一阈值设置 =====
IOU_THRESHOLD = 0.5  # IoU阈值：用于判断预测框与真实框是否匹配
EVAL_CONFIDENCE_THRESHOLD = 0.3  # 评估置信度阈值
# 评估阶段的一维NMS与温度缩放
NMS_IOU_THRESHOLD = 0.4  # 一维NMS的IoU阈值（类内去重）
EVAL_TEMPERATURE = 1.0   # 位置概率温度缩放（softmax 温度）

# 重要说明：
# 1. 类别一致性：严格要求预测类别=GT类别才能匹配
# 2. 背景类排除：背景类完全排除在TP判断外，从不参与评估
# 3. 置信度处理：标准mAP不使用固定置信度阈值预过滤，直接计算Precision-Recall曲线
# 4. 评估过滤：使用EVAL_CONFIDENCE_THRESHOLD过滤低质量预测框

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
训练和评估引擎模块
提供DETR模型的训练和评估功能

该模块包含两个主要函数：
1. train_one_epoch: 训练一个epoch
2. evaluate: 评估模型性能

主要功能：
- 训练循环管理
- 损失计算和反向传播
- 性能指标计算
- 模型评估和结果输出
"""


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    max_norm: float = 0,
                    lr_scheduler=None,
                    fold=None):
    """
    训练一个epoch
    """
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr',
                            utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error',
                            utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20

    # ===== 训练循环 =====
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # ===== 1. 数据预处理 =====
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # ===== 2. 前向传播 =====
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)

        # ===== 3. 反向传播 =====
        loss_value = losses.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        # ===== 4. 梯度裁剪 =====
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # ===== 5. 优化器步进 =====
        optimizer.step()

        # ===== 6. 指标更新 =====
        metric_logger.update(loss=loss_value, **
                             {k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(class_error=loss_dict['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # ===== 7. 返回训练指标 =====
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_1d_iou(box1, box2):
    """
    计算1D序列上的IoU
    box1, box2: [x0, x1] 格式，表示起始和结束位置
    """
    x0_1, x1_1 = box1
    x0_2, x1_2 = box2

    # 计算交集
    x0_i = max(x0_1, x0_2)
    x1_i = min(x1_1, x1_2)
    intersection = max(0, x1_i - x0_i)

    # 计算并集
    x0_u = min(x0_1, x0_2)
    x1_u = max(x1_1, x1_2)
    union = x1_u - x0_u

    # 计算IoU
    iou = intersection / union if union > 0 else 0.0
    return iou


def nms_1d_greedy(boxes, scores, iou_thresh):
    """
    对1D区间执行贪心NMS。
    boxes: Tensor [N, 2] （已在同一尺度）
    scores: Tensor [N]
    返回: 保留的索引（按score降序）
    """
    if boxes.numel() == 0:
        return []
    order = torch.argsort(scores, descending=True)
    keep = []
    suppressed = torch.zeros(len(order), dtype=torch.bool)
    for i in range(len(order)):
        if suppressed[i]:
            continue
        idx_i = order[i]
        keep.append(idx_i.item())
        bi = boxes[idx_i]
        for j in range(i + 1, len(order)):
            if suppressed[j]:
                continue
            idx_j = order[j]
            bj = boxes[idx_j]
            iou = compute_1d_iou(bi.tolist(), bj.tolist())
            if iou >= iou_thresh:
                suppressed[j] = True
    return keep


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             postprocessors: dict,
             data_loader: Iterable,
             device: torch.device,
             log_dir: str,
             epoch: int,
             fold: int = None,
             class_mapping: dict = None,
             localization_only: bool = False,
             sample_level_fp: bool = False,
             binary_mode: bool = False,
             bg_class_index: int = None):
    """
    评估模型性能

    评估流程：
    1. 模型设置为评估模式
    2. 对每个批次进行前向传播
    3. 后处理预测结果
    4. 使用贪心匹配计算mAP指标
    5. 保存结果（如果指定输出目录）

    Args:
        model: DETR模型
        criterion: 损失函数
        postprocessors: 后处理器
        data_loader: 数据加载器
        device: 计算设备
        log_dir: 输出目录
        epoch: 当前epoch数
        fold: 当前fold数
        class_mapping: 类别映射字典
        localization_only: 是否只计算定位任务
        sample_level_fp: 是否按样本级别计算假阳性（推荐用于负样本处理）
        binary_mode: 是否使用二分类模式
        bg_class_index: 背景类索引（仅在二分类模式下使用）

    Returns:
        dict: 包含评估指标的字典
    """
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 移除class_error meter，因为我们在评估时不使用训练时的class_error
    # metric_logger.add_meter('class_error',
    #                         utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Test:'

    # 在evaluate函数开头，兼容单GPU和多GPU环境获取num_classes
    if hasattr(model, 'module'):
        # 多GPU环境 (DataParallel/DistributedDataParallel)
        num_classes = model.module.class_embed.out_features
    else:
        # 单GPU环境
        num_classes = model.class_embed.out_features

    binary_mode = bool(binary_mode)
    if bg_class_index is None:
        bg_class_index = num_classes - 1

    detr_bg_class = bg_class_index

    # 现在类别映射保证no-object在最后，DETR背景类和数据集no-object类索引一致
    background_classes = [detr_bg_class]
    print(f"DETR背景类索引: {detr_bg_class} (no-object)")
    print(f"背景类列表: {background_classes}")
    print(f"模型类别总数: {num_classes}")

    # 初始化标签收集列表（只保留最终评估相关）
    collected_pred_labels = []
    collected_true_labels = []
    collected_pred_probs = []
    collected_pred_labels_bin = []
    collected_true_labels_bin = []
    collected_pred_probs_bin = []
    collected_pred_local = []
    # 样本级（二分类：是否存在BGC）
    collected_sample_labels = []  # 0/1
    collected_sample_scores = []  # 连续分数（用于AUC）
    collected_true_local = []

    # 全局分布统计收集器
    iou_distribution = []  # 收集所有预测框的最大IoU
    class_distribution = defaultdict(int)  # 收集预测类别分布
    bg_pred_count = 0  # 背景类预测计数
    non_bg_pred_count = 0  # 非背景类预测计数

    # 真实框信息收集
    gt_class_distribution = defaultdict(int)  # 收集真实类别分布
    gt_bg_count = 0  # 真实背景类计数
    gt_non_bg_count = 0  # 真实非背景类计数

    # ===== 贪心匹配mAP计算所需的数据结构 =====
    # 为每个类别收集预测和真实框信息
    # {class_id: [(score, pred_box, image_id, matched)]}
    all_predictions = defaultdict(list)
    all_targets = defaultdict(int)  # {class_id: total_gt_count}
    # {class_id: {image_id: [gt_boxes]}}
    all_gt_boxes_by_image = defaultdict(lambda: defaultdict(list))

    image_id = 0  # 图像索引

    # 统计变量
    total_predicted_boxes = 0
    total_target_boxes = 0
    total_correct_boxes = 0
    total_false_positive = 0
    total_false_negative = 0
    num_positive_samples = 0

    # 🔥 调试：FP来源统计
    fp_from_negative_samples = 0
    fp_from_positive_samples = 0

    # 调试变量
    debug_positive_images = 0
    debug_negative_images = 0
    debug_pred_boxes_collected = 0

    # ===== 新增：详细类别匹配分析 =====
    class_match_analysis = defaultdict(lambda: {
        'pred_count': 0,  # 该类别的预测框数量
        'gt_count': 0,  # 该类别的真实框数量
        'matched_count': 0,  # 成功匹配的数量
        'high_iou_count': 0,  # 高IoU但类别不匹配的数量
        'low_iou_count': 0,  # 低IoU的数量
    })

    # 跨类别高IoU统计
    # {pred_class: {gt_class: count}}
    cross_class_high_iou = defaultdict(lambda: defaultdict(int))

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # 记录损失
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        # 在评估阶段，我们不应该使用训练时的class_error
        # 而是应该计算评估时的分类准确率
        update_dict = {
            'loss': sum(loss_dict_reduced_scaled.values()) if loss_dict_reduced_scaled else 0.0
        }
        update_dict.update(loss_dict_reduced_scaled)
        update_dict.update(loss_dict_reduced_unscaled)
        metric_logger.update(**update_dict)

        # ===== 贪心匹配mAP计算 =====
        for i in range(len(outputs['pred_boxes'])):
            # 获取当前样本的CDS嵌入（用于精细化CDS级别预测）
            # samples 是 NestedTensor，需要通过 .tensors 属性访问
            sample_embedding = samples.tensors[i]  # [seq_len, embed_dim]
            seq_len = sample_embedding.shape[0]

            all_pred_boxes = outputs['pred_boxes'][i].cpu()
            all_pred_logits = outputs['pred_logits'][i].cpu()
            all_tgt_boxes = targets[i]['boxes'].cpu()
            all_tgt_labels = targets[i]['labels'].cpu()

            if binary_mode:
                bg_logits = all_pred_logits[..., detr_bg_class].unsqueeze(-1)
                if all_pred_logits.shape[-1] > 1:
                    pos_logits = torch.logsumexp(all_pred_logits[..., :detr_bg_class], dim=-1, keepdim=True)
                else:
                    pos_logits = torch.zeros_like(bg_logits)
                logits_binary = torch.cat([pos_logits, bg_logits], dim=-1)
                probs_binary = logits_binary.softmax(-1)
                labels_binary = logits_binary.argmax(-1)
                scores_binary = probs_binary[..., 0]

                tgt_binary = all_tgt_labels.clone()
                tgt_binary[tgt_binary != detr_bg_class] = 0
                tgt_binary[tgt_binary == detr_bg_class] = 1

                all_pred_labels = labels_binary
                all_pred_scores = scores_binary
            else:
                all_pred_labels = all_pred_logits.argmax(-1)
                all_pred_scores = all_pred_logits.softmax(-1).max(-1)[0]

            # 🔥 修复：统计每个类别的真实框数量
            for gt_label in all_tgt_labels:
                gt_label_item = gt_label.item() if hasattr(gt_label, 'item') else gt_label
                if gt_label_item not in background_classes:
                    class_match_analysis[gt_label_item]['gt_count'] += 1

            # ===== 调试：分析预测分布 =====
            if image_id == 0:  # 只在第一张图像时打印详细信息
                # [num_queries, num_classes]
                pred_probs = all_pred_logits.softmax(-1)
                print(f"\n===== 预测分布调试 (第一张图像) =====")
                print(f"预测logits shape: {all_pred_logits.shape}")
                print(f"预测概率 shape: {pred_probs.shape}")

                # 分析logits的原始值
                print(f"Logits统计:")
                print(f"  - 最小值: {all_pred_logits.min().item():.3f}")
                print(f"  - 最大值: {all_pred_logits.max().item():.3f}")
                print(f"  - 均值: {all_pred_logits.mean().item():.3f}")
                print(f"  - 标准差: {all_pred_logits.std().item():.3f}")

                # 打印query的概率分布
                print(f"前5个query的类别概率分布:")
                for q in range(min(5, len(pred_probs))):
                    probs = pred_probs[q]
                    logits = all_pred_logits[q]
                    pred_label = all_pred_labels[q]
                    max_prob = probs.max()
                    if hasattr(max_prob, 'item'):
                        max_prob = max_prob.item()
                    pred_label_val = pred_label
                    if hasattr(pred_label_val, 'item'):
                        pred_label_val = pred_label_val.item()
                    print(
                        f"  Query {q}: 预测类别={pred_label_val}, 最大概率={max_prob:.4f}")
                    print(
                        f"    Logits: {[f'{l:.3f}' for l in logits.tolist()]}")
                    print(
                        f"    Probs:  {[f'{p:.3f}' for p in probs.tolist()]}")

                # 统计各类别的平均预测概率
                mean_probs = pred_probs.mean(dim=0)
                print(
                    f"各类别平均预测概率: {[f'{p:.4f}' for p in mean_probs.tolist()]}")

                # 分析是否所有query都预测同一个类别
                unique_labels, counts = torch.unique(
                    all_pred_labels, return_counts=True)
                print(f"当前图像预测标签分布:")
                for label, count in zip(unique_labels, counts):
                    label_val = label
                    if hasattr(label_val, 'item'):
                        label_val = label_val.item()
                    count_val = count
                    if hasattr(count_val, 'item'):
                        count_val = count_val.item()
                    print(f"  类别 {label_val}: {count_val} 个预测框")

                # 检查分类头权重
                class_embed = model.module.class_embed if hasattr(
                    model, 'module') else model.class_embed
                print(
                    f"分类头偏置值: {[f'{b:.3f}' for b in class_embed.bias.tolist()]}")

                # ===== 新增：打印预测框位置和真实位置对比 =====
                print(f"\n===== 预测框位置和真实位置对比 =====")

                # 获取序列长度用于反归一化
                seq_len = targets[i].get('size', 128)
                if isinstance(seq_len, torch.Tensor):
                    if seq_len.numel() == 1:
                        seq_len = seq_len
                        if hasattr(seq_len, 'item'):
                            seq_len = seq_len.item()
                    else:
                        # 如果seq_len有多个元素，取第一个元素
                        seq_len_val = seq_len[0]
                        if hasattr(seq_len_val, 'item'):
                            seq_len_val = seq_len_val.item()
                        seq_len = seq_len_val

                # 打印前3个query的预测框位置
                print(f"前3个query的预测框位置:")
                for q in range(min(3, len(all_pred_boxes))):
                    pred_box = all_pred_boxes[q]
                    pred_label = all_pred_labels[q]
                    if hasattr(pred_label, 'item'):
                        pred_label = pred_label.item()
                    pred_score = all_pred_scores[q]
                    if hasattr(pred_score, 'item'):
                        pred_score = pred_score.item()

                    # 反归一化坐标
                    start_pos = int(pred_box[0].item() * seq_len)
                    end_pos = int(pred_box[1].item() * seq_len)

                    print(
                        f"  Query {q}: 预测类别={pred_label}, 置信度={pred_score:.4f}")
                    print(
                        f"    预测位置: [{pred_box[0].item():.3f}, {pred_box[1].item():.3f}] -> CDS[{start_pos}, {end_pos}]")

                # 打印所有真实框位置
                print(f"\n真实框位置:")
                if len(all_tgt_boxes) == 0:
                    print("  无真实框")
                else:
                    for gt_idx in range(len(all_tgt_boxes)):
                        gt_box = all_tgt_boxes[gt_idx]
                        gt_label = all_tgt_labels[gt_idx]
                        if hasattr(gt_label, 'item'):
                            gt_label = gt_label.item()

                        # 反归一化坐标
                        start_pos = int(gt_box[0].item() * seq_len)
                        end_pos = int(gt_box[1].item() * seq_len)

                        print(f"  真实框 {gt_idx}: 类别={gt_label}")
                        print(
                            f"    真实位置: [{gt_box[0].item():.3f}, {gt_box[1].item():.3f}] -> CDS[{start_pos}, {end_pos}]")

                # 计算并打印IoU对比
                print(f"\nIoU对比:")
                for q in range(min(3, len(all_pred_boxes))):
                    pred_box = all_pred_boxes[q]
                    pred_label = all_pred_labels[q].item()

                    best_iou = 0.0
                    best_gt_idx = -1

                    for gt_idx in range(len(all_tgt_boxes)):
                        gt_box = all_tgt_boxes[gt_idx]
                        gt_label = all_tgt_labels[gt_idx]
                        if hasattr(gt_label, 'item'):
                            gt_label = gt_label.item()

                        # 只计算同类别的IoU
                        if gt_label == pred_label:
                            iou = compute_1d_iou(
                                pred_box.tolist(), gt_box.tolist())
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx

                    if best_gt_idx >= 0:
                        gt_box = all_tgt_boxes[best_gt_idx]
                        print(
                            f"  Query {q} 与真实框 {best_gt_idx} (同类): IoU = {best_iou:.4f}")
                    else:
                        print(f"  Query {q}: 无同类真实框匹配")

                print("=" * 40)

            # 判断是否为正样本图像（有非背景类目标）
            if binary_mode:
                non_bg_gt_mask = [label == 0 for label in tgt_binary]
            else:
                non_bg_gt_mask = [label not in background_classes for label in all_tgt_labels]
            is_positive_image = any(non_bg_gt_mask)

            # 样本级分数改为定位口径：后续根据"是否存在匹配成功的预测框"聚合

            if is_positive_image:
                num_positive_samples += 1
                debug_positive_images += 1
            else:
                debug_negative_images += 1

            # ===== 按类别进行贪心匹配 =====
            # 🔥 修改说明：对于正样本图像，只关注匹配到的预测框
            # 未匹配的预测框不计入FP，避免假阳性过高
            # 1. 统计每个类别的真实框数量并收集真实框信息
            for gt_idx, gt_label in enumerate(all_tgt_labels):
                gt_label = gt_label.item() if hasattr(gt_label, 'item') else gt_label
                gt_box = all_tgt_boxes[gt_idx]

                if gt_label not in background_classes:
                    all_targets[gt_label] += 1
                    total_target_boxes += 1
                    # 收集该类别的真实框到对应图像
                    all_gt_boxes_by_image[gt_label][image_id].append(gt_box)

                # 收集真实框分布统计
                gt_class_distribution[gt_label] += 1
                if gt_label in background_classes:
                    gt_bg_count += 1
                else:
                    gt_non_bg_count += 1

            # 2. 收集每个类别的预测框（添加置信度阈值过滤）
            for pred_idx in range(len(all_pred_boxes)):
                pred_label = all_pred_labels[pred_idx].item()
                pred_score = all_pred_scores[pred_idx].item()
                pred_box = all_pred_boxes[pred_idx]

                # 收集预测分布统计
                class_distribution[pred_label] += 1
                if pred_label in background_classes:
                    bg_pred_count += 1
                else:
                    non_bg_pred_count += 1

                # ===== 置信度过滤：只有高置信度的非背景类预测框才参与评估 =====
                if (pred_label not in background_classes and
                        pred_score >= EVAL_CONFIDENCE_THRESHOLD):

                    total_predicted_boxes += 1  # 统计通过置信度过滤的预测框
                    debug_pred_boxes_collected += 1

                    # 计算该预测框与同类别真实框的最大IoU（用于分布统计）
                    max_iou = 0.0
                    for gt_idx, gt_box in enumerate(all_tgt_boxes):
                        gt_label = all_tgt_labels[gt_idx]
                        if hasattr(gt_label, 'item'):
                            gt_label = gt_label.item()

                        # 只计算同类别的IoU
                        if gt_label == pred_label:
                            iou = compute_1d_iou(
                                pred_box.tolist(), gt_box.tolist())
                            if iou > max_iou:
                                max_iou = iou
                    iou_distribution.append(max_iou)

                    # 添加到该类别的预测列表中 (score, pred_box, image_id, matched)
                    all_predictions[pred_label].append(
                        (pred_score, pred_box, image_id, False))

            # 3. 对当前图像进行贪心匹配（用于位置和分类指标收集）
            if is_positive_image:
                # 分类与定位指标收集初始化
                image_pred_labels = []  # 预测类别
                image_true_labels = []  # 真实类别
                image_pred_probs = []  # 概率分布
                image_loc_labels = []  # 定位正确标签
                image_loc_scores = []  # 定位置信度

                # [num_queries, num_classes]
                pred_probs = all_pred_logits.softmax(-1)

                # 对当前图像进行贪心匹配来收集分类和定位指标
                num_queries = len(all_pred_boxes)
                true_labels = torch.full((num_queries,), background_classes[0])
                loc_correct = torch.zeros(num_queries)

                # 创建匹配状态
                gt_matched = [False] * len(all_tgt_boxes)
                pred_matched = [False] * num_queries  # 新增：记录预测框是否被匹配

                # 按置信度排序所有预测框（不使用置信度阈值过滤）
                sorted_indices = torch.argsort(
                    all_pred_scores, descending=True)

                for pred_idx in sorted_indices:
                    pred_box = all_pred_boxes[pred_idx]
                    pred_label = all_pred_labels[pred_idx]
                    if hasattr(pred_label, 'item'):
                        pred_label = pred_label.item()
                    pred_score = all_pred_scores[pred_idx]
                    if hasattr(pred_score, 'item'):
                        pred_score = pred_score.item()

                    # ===== 置信度过滤：只处理高置信度的预测框 =====
                    if (pred_label not in background_classes and
                            pred_score < EVAL_CONFIDENCE_THRESHOLD):
                        continue  # 跳过低置信度预测框

                    best_iou = 0.0
                    best_gt_idx = -1
                    best_gt_label = -1

                    # ===== 详细匹配分析：检查与所有GT框的IoU =====
                    for gt_idx in range(len(all_tgt_boxes)):
                        if gt_matched[gt_idx]:
                            continue

                        gt_label = all_tgt_labels[gt_idx]
                        if hasattr(gt_label, 'item'):
                            gt_label = gt_label.item()
                        gt_box = all_tgt_boxes[gt_idx]

                        iou = compute_1d_iou(
                            pred_box.tolist(), gt_box.tolist())

                        # 记录最佳IoU匹配（不考虑类别）
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                            best_gt_label = gt_label

                    # ===== 统计类别匹配分析 =====
                    if pred_label not in background_classes:
                        class_match_analysis[pred_label]['pred_count'] += 1

                        if best_iou >= IOU_THRESHOLD and best_gt_idx != -1:
                            if localization_only or pred_label == best_gt_label:
                                # 定位任务模式：只考虑IoU，不考虑类别
                                # 或者同类别且高IoU：成功匹配
                                # 🔍 只打印第一对成功匹配的信息
                                if total_correct_boxes == 0:  # 第一次TP匹配
                                    pred_class_name = class_mapping.get(
                                        pred_label, f"类别{pred_label}") if class_mapping else f"类别{pred_label}"
                                    gt_class_name = class_mapping.get(
                                        best_gt_label, f"类别{best_gt_label}") if class_mapping else f"类别{best_gt_label}"
                                    if localization_only:
                                        print(f"\n🔍 定位任务成功匹配示例:")
                                        print(
                                            f"  预测框: {pred_class_name}, 位置[{pred_box[0].item():.3f}, {pred_box[1].item():.3f}]")
                                        print(
                                            f"  真实框: {gt_class_name}, 位置[{all_tgt_boxes[best_gt_idx][0].item():.3f}, {all_tgt_boxes[best_gt_idx][1].item():.3f}]")
                                        print(f"  IoU: {best_iou:.3f} (忽略类别)")
                                    else:
                                        print(f"\n🔍 成功匹配示例:")
                                        print(
                                            f"  预测框: {pred_class_name}, 位置[{pred_box[0].item():.3f}, {pred_box[1].item():.3f}]")
                                        print(
                                            f"  真实框: {gt_class_name}, 位置[{all_tgt_boxes[best_gt_idx][0].item():.3f}, {all_tgt_boxes[best_gt_idx][1].item():.3f}]")
                                        print(f"  IoU: {best_iou:.3f}")

                                class_match_analysis[pred_label]['matched_count'] += 1
                                gt_matched[best_gt_idx] = True
                                pred_matched[pred_idx] = True  # 标记预测框已匹配
                                true_labels[pred_idx] = all_tgt_labels[best_gt_idx]
                                loc_correct[pred_idx] = 1
                                total_correct_boxes += 1
                            else:
                                # 不同类别但高IoU：记录跨类别匹配
                                class_match_analysis[pred_label]['high_iou_count'] += 1
                                cross_class_high_iou[pred_label][best_gt_label] += 1
                                # 在定位任务模式下，高IoU但类别不匹配也算作假阳性
                                if not localization_only:
                                    total_false_positive += 1
                                    fp_from_positive_samples += 1
                        else:
                            # 低IoU：记录低IoU情况，但不增加FP（只考虑匹配的框）
                            class_match_analysis[pred_label]['low_iou_count'] += 1
                            # 🔥 修改：对于正样本，未匹配的预测框不计入FP
                            # 移除这行：total_false_positive += 1

                # 统计未匹配的真实框（非背景类）为FN
                for gt_idx in range(len(all_tgt_boxes)):
                    if not gt_matched[gt_idx] and all_tgt_labels[gt_idx].item() not in background_classes:
                        total_false_negative += 1

                # ===== 样本级分数（与定位口径一致）：存在任一匹配成功的预测框即为正，分数取匹配框最高分 =====
                matched_indices = [idx for idx,
                                   m in enumerate(pred_matched) if m]
                if len(matched_indices) > 0:
                    best_score = float(
                        max([all_pred_scores[idx].item() for idx in matched_indices]))
                else:
                    best_score = 0.0
                collected_sample_labels.append(1)
                collected_sample_scores.append(best_score)

                # ===== 🔥 修正：收集匹配框的分类指标数据 =====
                # 收集所有IoU满足阈值的匹配框的分类数据（包括类别正确和错误的）
                if not localization_only:  # 只在非定位任务模式下收集分类指标
                    for pred_idx in range(num_queries):
                        pred_label = all_pred_labels[pred_idx].item()
                        pred_score = all_pred_scores[pred_idx].item()

                        # 收集所有IoU满足阈值的预测框
                        if (pred_label not in background_classes and
                                pred_score >= EVAL_CONFIDENCE_THRESHOLD):

                            # 检查是否与任何GT框匹配（IoU满足阈值）
                            best_iou = 0.0
                            best_gt_label = -1

                            for gt_idx in range(len(all_tgt_boxes)):
                                gt_label = all_tgt_labels[gt_idx]
                                if hasattr(gt_label, 'item'):
                                    gt_label = gt_label.item()
                                if gt_label not in background_classes:
                                    gt_box = all_tgt_boxes[gt_idx]
                                    pred_box = all_pred_boxes[pred_idx]
                                    iou = compute_1d_iou(
                                        pred_box.tolist(), gt_box.tolist())

                                    if iou >= IOU_THRESHOLD and iou > best_iou:
                                        best_iou = iou
                                        best_gt_label = gt_label

                            # 如果找到匹配的GT框，收集分类数据
                            if best_gt_label != -1:
                                image_pred_labels.append(pred_label)
                                image_pred_probs.append(
                                    pred_probs[pred_idx].tolist())
                                image_true_labels.append(best_gt_label)

                if binary_mode and not localization_only:
                    pos_probs_queries = (1.0 - pred_probs[:, detr_bg_class])
                    bg_probs_queries = pred_probs[:, detr_bg_class]
                    prob_vectors = torch.stack([pos_probs_queries, bg_probs_queries], dim=-1)
                    pred_labels_bin = (pos_probs_queries >= 0.5).long()
                    true_labels_bin = (true_labels != detr_bg_class).long()

                    collected_pred_probs_bin.extend(prob_vectors.tolist())
                    collected_pred_labels_bin.extend(pred_labels_bin.tolist())
                    collected_true_labels_bin.extend(true_labels_bin.tolist())

                # ===== 定位标签收集（只收集正样本图像） =====
                # 获取序列长度用于反归一化
                seq_len = targets[i].get('size', 128)
                if isinstance(seq_len, torch.Tensor):
                    if seq_len.numel() == 1:
                        seq_len = seq_len
                        if hasattr(seq_len, 'item'):
                            seq_len = seq_len.item()
                    else:
                        seq_len_val = seq_len[0]
                        if hasattr(seq_len_val, 'item'):
                            seq_len_val = seq_len_val.item()
                        seq_len = seq_len_val

                # 创建CDS级别的标签和置信度
                cds_length = seq_len
                cds_true_labels = torch.zeros(
                    cds_length, dtype=torch.bool)  # 真实CDS标签
                cds_pred_scores = torch.zeros(
                    cds_length, dtype=torch.float)  # 预测CDS置信度

                # 1. 标记真实BGC区域的CDS位置
                for gt_idx in range(len(all_tgt_boxes)):
                    gt_label = all_tgt_labels[gt_idx]
                    if hasattr(gt_label, 'item'):
                        gt_label = gt_label.item()
                    if gt_label not in background_classes:  # 只处理非背景类
                        gt_box = all_tgt_boxes[gt_idx]
                        # 反归一化坐标
                        start_pos = int(gt_box[0].item() * seq_len)
                        end_pos = int(gt_box[1].item() * seq_len)
                        # 确保位置在有效范围内
                        start_pos = max(0, min(start_pos, cds_length - 1))
                        end_pos = max(start_pos, min(end_pos, cds_length))
                        # 标记真实BGC区域的CDS为True
                        cds_true_labels[start_pos:end_pos] = True

                # 2. 计算预测BGC区域的CDS概率（基于框的联合逻辑）
                    base_prob = 0.0
                    cds_pred_scores.fill_(base_prob)
                    if EVAL_TEMPERATURE != 1.0:
                        pred_probs_full = (
                            all_pred_logits / EVAL_TEMPERATURE).softmax(-1)
                    else:
                        pred_probs_full = all_pred_logits.softmax(-1)
                    bg_index = background_classes[0]
                    pos_probs = 1.0 - pred_probs_full[:, bg_index]
                    sorted_pred_indices = torch.argsort(
                        pos_probs, descending=True)
                    all_boxes_abs = (all_pred_boxes *
                                     seq_len).clamp(0, seq_len).cpu()
                    keep_indices = nms_1d_greedy(
                        all_boxes_abs, pos_probs.cpu(), NMS_IOU_THRESHOLD)
                    keep_set = set(keep_indices)
                    for pred_idx in sorted_pred_indices:
                        if pred_idx.item() not in keep_set:
                            continue
                        pred_score = pos_probs[pred_idx]
                        if hasattr(pred_score, 'item'):
                            pred_score = pred_score.item()
                        if pred_score >= EVAL_CONFIDENCE_THRESHOLD:
                            pred_box = all_pred_boxes[pred_idx]
                            start_pos = int(pred_box[0].item() * seq_len)
                            end_pos = int(pred_box[1].item() * seq_len)
                            start_pos = max(0, min(start_pos, cds_length - 1))
                            end_pos = max(start_pos, min(end_pos, cds_length))
                            center = (start_pos + end_pos) / 2
                            box_length = max(1, end_pos - start_pos)
                            if end_pos > start_pos:
                                positions = torch.arange(start_pos, end_pos)
                                distances = (positions - center).abs().float()
                                sigma = max(1.0, (box_length / 2.0) + 3.0)
                                distance_weight = torch.exp(
                                    - (distances ** 2) / (2.0 * (sigma ** 2)))
                                pred_probs = float(
                                    pred_score) * distance_weight
                                old = cds_pred_scores[positions].float()
                                cds_pred_scores[positions] = 1.0 - \
                                    (1.0 - old) * (1.0 - pred_probs)

                # 3. 收集CDS级别的标签和置信度
                for cds_pos in range(cds_length):
                    image_loc_labels.append(
                        int(cds_true_labels[cds_pos].item()))
                    image_loc_scores.append(cds_pred_scores[cds_pos].item())

                # 全局收集
                collected_pred_labels.extend(image_pred_labels)
                collected_true_labels.extend(image_true_labels)
                collected_pred_probs.extend(image_pred_probs)
                collected_true_local.extend(image_loc_labels)
                collected_pred_local.extend(image_loc_scores)
            else:
                # ===== 负样本图像处理（修改FP计算逻辑） =====
                # 🔥 修改：负样本中只要有预测正样本BGC类别就算一个FP
                has_positive_prediction = False  # 标记是否有正样本预测

                for pred_idx in range(len(all_pred_boxes)):
                    pred_label = all_pred_labels[pred_idx]
                    if hasattr(pred_label, 'item'):
                        pred_label = pred_label.item()
                    pred_score = all_pred_scores[pred_idx]
                    if hasattr(pred_score, 'item'):
                        pred_score = pred_score.item()

                    # 检查是否有预测正样本BGC类别（加入置信度阈值过滤）
                    if (pred_label not in background_classes and
                            pred_score >= EVAL_CONFIDENCE_THRESHOLD):
                        has_positive_prediction = True
                        break  # 找到一个就够了，不需要继续检查

                # 样本级分数：无匹配，记为0
                collected_sample_labels.append(0)
                collected_sample_scores.append(0.0)

                # 每个负样本最多贡献1个FP
                if has_positive_prediction:
                    total_false_positive += 1
                    fp_from_negative_samples += 1

                if binary_mode and not localization_only:
                    pred_probs = all_pred_logits.softmax(-1)
                    pos_probs_queries = (1.0 - pred_probs[:, detr_bg_class])
                    bg_probs_queries = pred_probs[:, detr_bg_class]
                    prob_vectors = torch.stack([pos_probs_queries, bg_probs_queries], dim=-1)
                    pred_labels_bin = (pos_probs_queries >= 0.5).long()
                    true_labels_bin = torch.zeros_like(pred_labels_bin)

                    collected_pred_probs_bin.extend(prob_vectors.tolist())
                    collected_pred_labels_bin.extend(pred_labels_bin.tolist())
                    collected_true_labels_bin.extend(true_labels_bin.tolist())

            image_id += 1

    # ===== 计算每个类别的AP（贪心匹配） =====
    class_aps = {}
    class_precisions = {}
    class_recalls = {}

    for class_id in all_predictions.keys():
        if class_id in background_classes:
            continue

        # 获取该类别的所有预测框，按置信度降序排序
        class_predictions = all_predictions[class_id]
        class_predictions.sort(key=lambda x: x[0], reverse=True)  # 按score降序排序

        # 获取该类别的真实框总数
        total_gt = all_targets[class_id]

        if total_gt == 0:
            continue

        print(f"\n类别 {class_id}: 预测框数={len(class_predictions)}, 真实框数={total_gt}")

        # 获取该类别的真实框信息
        gt_boxes_by_image = all_gt_boxes_by_image[class_id]

        # 执行贪心匹配（标准mAP计算流程）
        tp = []
        fp = []
        scores = []

        # 为每个图像创建真实框匹配状态
        gt_matched_by_image = {}
        for img_id, gt_boxes in gt_boxes_by_image.items():
            gt_matched_by_image[img_id] = [False] * len(gt_boxes)

        # 按置信度从高到低处理每个预测框（标准mAP：不使用置信度阈值过滤）
        for score, pred_box, img_id, _ in class_predictions:
            scores.append(score)

            # 在对应图像中寻找最佳匹配的真实框
            # 注意：由于是按类别分别计算，gt_boxes_by_image[class_id]中的所有GT框
            # 都已经是当前class_id类别，保证了严格的类别一致性匹配
            best_iou = 0.0
            best_gt_idx = -1

            if img_id in gt_boxes_by_image:
                for gt_idx, gt_box in enumerate(gt_boxes_by_image[img_id]):
                    if gt_matched_by_image[img_id][gt_idx]:
                        continue

                    # 计算IoU（类别已经通过数据结构保证一致）
                    iou = compute_1d_iou(pred_box.tolist(), gt_box.tolist())
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

            # 使用IoU阈值判断TP/FP（这里是mAP标准中的唯一阈值）
            if best_iou >= IOU_THRESHOLD and best_gt_idx != -1:
                tp.append(1)  # True Positive: 相同类别且IoU≥阈值
                fp.append(0)
                gt_matched_by_image[img_id][best_gt_idx] = True
            else:
                tp.append(0)
                fp.append(1)  # False Positive: 未找到匹配或IoU<阈值

        # 计算precision和recall
        tp = np.array(tp)
        fp = np.array(fp)
        scores = np.array(scores)

        # 累积计算
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # 计算precision和recall
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / (total_gt + 1e-6)

        # 计算AP (使用11点插值)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11

        class_aps[class_id] = ap
        class_precisions[class_id] = precisions[-1] if len(
            precisions) > 0 else 0
        class_recalls[class_id] = recalls[-1] if len(recalls) > 0 else 0

        class_name = class_mapping.get(
            class_id, f"类别{class_id}") if class_mapping else f"类别{class_id}"
        print(
            f"  {class_name}: AP={ap:.4f}, P={precisions[-1]:.4f}, R={recalls[-1]:.4f}")

    # 计算mAP
    mean_ap = np.mean(list(class_aps.values())) if class_aps else 0.0
    mean_precision = np.mean(
        list(class_precisions.values())) if class_precisions else 0.0
    mean_recall = np.mean(list(class_recalls.values())
                          ) if class_recalls else 0.0
    map_f1 = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall) if (
        mean_precision + mean_recall) > 0 else 0.0

    print(
        f"\n总体mAP结果: mAP={mean_ap:.4f}, mP={mean_precision:.4f}, mR={mean_recall:.4f}, F1={map_f1:.4f}")

    # 计算位置精度评估指标
    precision = total_correct_boxes / (total_correct_boxes + total_false_positive) if (
        total_correct_boxes + total_false_positive) > 0 else 0
    recall = total_correct_boxes / (total_correct_boxes + total_false_negative) if (
        total_correct_boxes + total_false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0

    # 打印全局分布统计
    print("\n===== 全局分布统计 =====")
    print(f"正样本数量: {num_positive_samples}")

    # 1. IoU分布分析
    if len(iou_distribution) > 0:
        ious = np.array(iou_distribution, dtype=np.float32)  # 明确指定数据类型
        print(f"\nIoU分布 (共{len(ious)}个预测框):")
        print(f"  - 平均IoU: {ious.mean():.4f}")
        print(
            f"  - IoU < 0.3: {(ious < 0.3).sum()} ({(ious < 0.3).sum() / len(ious) * 100:.1f}%)")
        print(
            f"  - 0.3 ≤ IoU < 0.5: {((ious >= 0.3) & (ious < 0.5)).sum()} ({((ious >= 0.3) & (ious < 0.5)).sum() / len(ious) * 100:.1f}%)")
        print(
            f"  - IoU ≥ 0.5: {(ious >= 0.5).sum()} ({(ious >= 0.5).sum() / len(ious) * 100:.1f}%)")
    else:
        print("\n无IoU数据可统计（可能没有正样本）")

    # 2. 预测类别分布分析
    if bg_pred_count + non_bg_pred_count > 0:
        total_pred = bg_pred_count + non_bg_pred_count
        print(f"\n预测类别分布 (共{total_pred}个预测框):")
        print(
            f"  - 背景类预测: {bg_pred_count} ({bg_pred_count / total_pred * 100:.1f}%)")
        print(
            f"  - 非背景类预测: {non_bg_pred_count} ({non_bg_pred_count / total_pred * 100:.1f}%)")

        print("\n预测各类别数量:")
        for cls, count in sorted(class_distribution.items()):
            if cls in background_classes:
                cls_name = "背景(no-object)"
            elif class_mapping and cls in class_mapping:
                cls_name = f"{class_mapping[cls]}(索引{cls})"
            else:
                cls_name = f"类别{cls}"
            print(f"  - {cls_name}: {count}个")
    else:
        print("\n无预测类别分布数据可统计")

    # 3. 真实类别分布分析
    if gt_bg_count + gt_non_bg_count > 0:
        total_gt = gt_bg_count + gt_non_bg_count
        print(f"\n真实类别分布 (共{total_gt}个真实框):")
        print(
            f"  - 背景类真实框: {gt_bg_count} ({gt_bg_count / total_gt * 100:.1f}%)")
        print(
            f"  - 非背景类真实框: {gt_non_bg_count} ({gt_non_bg_count / total_gt * 100:.1f}%)")

        print("\n真实各类别数量:")
        for cls, count in sorted(gt_class_distribution.items()):
            if cls in background_classes:
                cls_name = "背景(no-object)"
            elif class_mapping and cls in class_mapping:
                cls_name = f"{class_mapping[cls]}(索引{cls})"
            else:
                cls_name = f"类别{cls}"
            print(f"  - {cls_name}: {count}个")
    else:
        print("\n无真实类别分布数据可统计")

    print("=" * 50)

    # ===== 置信度过滤效果分析 =====
    print(f"\n🎯 置信度过滤效果分析 (阈值={EVAL_CONFIDENCE_THRESHOLD}):")
    filtered_pred_ratio = (
        non_bg_pred_count - total_predicted_boxes) / non_bg_pred_count if non_bg_pred_count > 0 else 0
    print(f"  • 原始非背景预测框: {non_bg_pred_count:,}个")
    print(f"  • 通过置信度过滤: {total_predicted_boxes:,}个")
    print(f"  • 被过滤掉的预测框: {non_bg_pred_count - total_predicted_boxes:,}个")
    print(f"  • 过滤比例: {filtered_pred_ratio:.1%}")

    if total_predicted_boxes > 0:
        filtered_precision = total_correct_boxes / total_predicted_boxes
        print(f"  • 过滤后精确率: {filtered_precision:.1%}")
        print(f"  • 精确率提升效果: 显著减少了低质量预测框")

    # ===== 详细类别匹配分析报告 =====
    print(f"\n🔍 详细类别匹配分析报告 (置信度≥{EVAL_CONFIDENCE_THRESHOLD}):")
    print(f"{'类别':<12} {'预测框':<8} {'真实框':<8} {'成功匹配':<10} {'匹配率':<10} {'高IoU失配':<12} {'低IoU':<8}")
    print("-" * 80)

    total_pred_analyzed = 0
    total_gt_analyzed = 0
    total_matched_analyzed = 0
    total_high_iou_mismatch = 0
    total_low_iou = 0

    for class_id in sorted(set(list(class_match_analysis.keys()) + list(gt_class_distribution.keys()))):
        if class_id in background_classes:
            continue

        analysis = class_match_analysis[class_id]
        pred_count = analysis['pred_count']
        gt_count = analysis['gt_count']
        matched_count = analysis['matched_count']
        high_iou_count = analysis['high_iou_count']
        low_iou_count = analysis['low_iou_count']

        match_rate = matched_count / pred_count if pred_count > 0 else 0

        class_name = class_mapping.get(
            class_id, f"类别{class_id}") if class_mapping else f"类别{class_id}"
        print(
            f"{class_name:<12} {pred_count:<8} {gt_count:<8} {matched_count:<10} {match_rate:<10.3f} {high_iou_count:<12} {low_iou_count:<8}")

        total_pred_analyzed += pred_count
        total_gt_analyzed += gt_count
        total_matched_analyzed += matched_count
        total_high_iou_mismatch += high_iou_count
        total_low_iou += low_iou_count

    print("-" * 80)
    overall_match_rate = total_matched_analyzed / \
        total_pred_analyzed if total_pred_analyzed > 0 else 0
    print(
        f"{'总计':<12} {total_pred_analyzed:<8} {total_gt_analyzed:<8} {total_matched_analyzed:<10} {overall_match_rate:<10.3f} {total_high_iou_mismatch:<12} {total_low_iou:<8}")

    print(f"\n📊 关键发现:")
    print(f"  • 高置信度预测框: {total_pred_analyzed:,}")
    print(f"  • 成功匹配: {total_matched_analyzed} ({overall_match_rate:.1%})")

    # 避免除零错误
    if total_pred_analyzed > 0:
        print(
            f"  • 高IoU但类别不匹配: {total_high_iou_mismatch} ({total_high_iou_mismatch / total_pred_analyzed:.1%})")
        print(
            f"  • 低IoU: {total_low_iou} ({total_low_iou / total_pred_analyzed:.1%})")
    else:
        print(f"  • 高IoU但类别不匹配: {total_high_iou_mismatch} (0.0%)")
        print(f"  • 低IoU: {total_low_iou} (0.0%)")

    # 分类F1/AUC
    debug_info = {}  # 用于收集DEBUG信息
    # 统一标签变量名，确保后续指标计算使用正确数据（只用自定义收集的变量）
    if binary_mode and not localization_only:
        all_true_labels = collected_true_labels_bin
        all_pred_labels = collected_pred_labels_bin
        all_pred_probs = collected_pred_probs_bin
    else:
        all_true_labels = collected_true_labels
        all_pred_labels = collected_pred_labels
        all_pred_probs = collected_pred_probs
    all_true_local = collected_true_local
    all_pred_local = collected_pred_local

    # 调试信息打印
    print(f"\n===== 调试信息 =====")
    print(f"正样本图像数: {debug_positive_images}")
    print(f"负样本图像数: {debug_negative_images}")
    print(f"总图像数: {debug_positive_images + debug_negative_images}")
    print(f"实际收集的预测框数: {debug_pred_boxes_collected}")
    print(f"total_predicted_boxes: {total_predicted_boxes}")
    print(f"非背景类预测框总数: {non_bg_pred_count}")
    print(f"背景类索引: {background_classes}")

    # 分析预测类别分布
    print(f"\n===== 预测类别分布分析 =====")
    total_predictions = sum(class_distribution.values())
    print(f"所有预测框总数: {total_predictions}")
    print("各类别预测占比:")
    for cls in sorted(class_distribution.keys()):
        count = class_distribution[cls]
        percentage = count / total_predictions * 100 if total_predictions > 0 else 0
        cls_name = f"背景(no-object)" if cls in background_classes else f"类别{cls}"
        print(f"  {cls_name}: {count}个 ({percentage:.1f}%)")

    # 分析真实类别分布
    print(f"\n===== 真实类别分布分析 =====")
    total_gt = sum(gt_class_distribution.values())
    print(f"所有真实框总数: {total_gt}")
    print("各类别真实占比:")
    for cls in sorted(gt_class_distribution.keys()):
        count = gt_class_distribution[cls]
        percentage = count / total_gt * 100 if total_gt > 0 else 0
        cls_name = f"背景(no-object)" if cls in background_classes else f"类别{cls}"
        print(f"  {cls_name}: {count}个 ({percentage:.1f}%)")

    print("=" * 25)

    # 只在非定位任务模式下计算分类指标
    if not localization_only:
        try:
            # 抑制NumPy警告
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                # 计算每个类别的F1分数
                class_f1_macro = f1_score(
                    all_true_labels, all_pred_labels, average='macro', zero_division=0)
                class_f1_weighted = f1_score(
                    all_true_labels, all_pred_labels, average='weighted', zero_division=0)
                class_f1_per_class = f1_score(
                    all_true_labels, all_pred_labels, average=None, zero_division=0).tolist()

                # 检查all_pred_probs的shape
                all_pred_probs_np = np.array(all_pred_probs)
                if all_pred_probs_np.ndim == 2 and all_pred_probs_np.shape[0] == len(all_true_labels):
                    # 检查每个类别的样本数
                    class_counts = np.bincount(all_true_labels)
                    valid_classes = np.where(class_counts > 0)[0]

                    if len(valid_classes) > 1:
                        # 确保标签是one-hot形式
                        y_true_one_hot = np.eye(all_pred_probs_np.shape[1])[
                            all_true_labels]
                        # 只使用有样本的类别计算AUC
                        try:
                            class_auc = roc_auc_score(
                                y_true_one_hot[:, valid_classes],
                                all_pred_probs_np[:, valid_classes],
                                average='macro',
                                multi_class='ovr'
                            )
                            # print(f"[DEBUG] 分类AUC: {class_auc:.4f}")
                        except Exception as auc_error:
                            # print(f"[DEBUG] AUC计算错误: {str(auc_error)}")
                            class_auc = 0.0
                    else:
                        print(
                            f'[DEBUG] 警告: 只有一个类别({valid_classes[0]})，无法计算AUC')
                        debug_info['auc_warning'] = f'只有一个类别({valid_classes[0]})，无法计算AUC'
                        class_auc = 0.0
                else:
                    print(
                        f'[DEBUG] 警告: all_pred_probs shape不正确: {all_pred_probs_np.shape}, true_label数: {len(all_true_labels)}')
                    debug_info[
                        'shape_warning'] = f'all_pred_probs shape不正确: {all_pred_probs_np.shape}, true_label数: {len(all_true_labels)}'
                    class_auc = 0.0
                class_confusion = confusion_matrix(
                    all_true_labels, all_pred_labels).tolist()
        except Exception as e:
            debug_info['calculation_error'] = str(e)
            class_f1_macro = 0.0
            class_f1_weighted = 0.0
            class_f1_per_class = []
            class_auc = 0.0
            class_confusion = []
    else:
        # 定位任务模式：设置默认值
        class_f1_macro = 0.0
        class_f1_weighted = 0.0
        class_f1_per_class = []
        class_auc = 0.0
        class_confusion = []
        print("[DEBUG] 定位任务模式：跳过分类指标计算")

    # ===== 样本级（二分类）AUC/F1 =====
    try:
        if len(collected_sample_labels) == len(collected_sample_scores) and len(collected_sample_labels) > 0:
            # 选择阈值0.5计算样本级F1；AUC使用连续分数
            sample_pred_labels = [
                1 if s >= 0.5 else 0 for s in collected_sample_scores]
            sample_f1 = f1_score(
                collected_sample_labels, sample_pred_labels, average='binary', zero_division=0)
            if len(set(collected_sample_labels)) > 1:
                sample_auc = roc_auc_score(
                    collected_sample_labels, collected_sample_scores)
            else:
                sample_auc = 0.5
        else:
            sample_f1, sample_auc = 0.0, 0.5
    except Exception:
        sample_f1, sample_auc = 0.0, 0.5

    # ===== 位置F1/AUC计算（CDS级别） =====
    try:
        # 计算位置F1分数
        if len(all_true_local) > 0 and len(all_pred_local) > 0:
            # 添加调试信息
            print(f"CDS级别定位评估:")
            print(f"  总CDS数: {len(all_true_local)}")
            print(f"  真实BGC区域CDS数: {sum(all_true_local)}")
            print(
                f"  预测BGC区域CDS数: {sum([1 for x in all_pred_local if x > 0.5])}")
            print(
                f"  预测标签分布: 0={sum([1 for x in all_pred_local if x < 0.5])}, 1={sum([1 for x in all_pred_local if x > 0.5])}")

            # 🔧 修复：为F1计算创建二进制预测标签
            loc_pred_labels = [1 if prob >
                               0.5 else 0 for prob in all_pred_local]

            # 计算CDS级别的TP、FP、FN
            cds_tp = sum(1 for pred, true in zip(loc_pred_labels,
                         all_true_local) if pred == 1 and true == 1)
            cds_fp = sum(1 for pred, true in zip(loc_pred_labels,
                         all_true_local) if pred == 1 and true == 0)
            cds_fn = sum(1 for pred, true in zip(loc_pred_labels,
                         all_true_local) if pred == 0 and true == 1)

            # 计算CDS级别的精确率、召回率、F1
            cds_precision = cds_tp / \
                (cds_tp + cds_fp) if (cds_tp + cds_fp) > 0 else 0
            cds_recall = cds_tp / \
                (cds_tp + cds_fn) if (cds_tp + cds_fn) > 0 else 0
            loc_f1 = f1_score(all_true_local, loc_pred_labels,
                              average='binary', zero_division=0)

            print(f"  CDS级别精确率: {cds_precision:.4f}")
            print(f"  CDS级别召回率: {cds_recall:.4f}")
            print(f"  CDS级别F1: {loc_f1:.4f}")
            print(f"  CDS级别统计: TP={cds_tp}, FP={cds_fp}, FN={cds_fn}")

            # 计算CDS级别AUC（优化：使用连续预测值）
            if len(set(all_true_local)) > 1:  # 确保有正负样本
                try:
                    # 直接使用连续预测值计算AUC（不再转换为二进制）
                    # 这样能更好地反映预测的质量和置信度
                    loc_auc = roc_auc_score(all_true_local, all_pred_local)
                    print(f"  CDS级别AUC: {loc_auc:.4f}")

                except Exception as e:
                    print(f"  [警告] AUC计算失败: {e}")
                    loc_auc = 0.5  # 默认值
            else:
                loc_auc = 0.5  # 只有一种标签时返回默认值
                print(f"  [警告] 只有一种标签，无法计算有意义的AUC (AUC={loc_auc:.4f})")
        else:
            loc_f1 = 0.0
            loc_auc = 0.0
            print(f"[DEBUG] 位置指标警告: 没有收集到位置数据")
    except Exception as e:
        print(f"[DEBUG] 位置指标计算错误: {e}")
        loc_f1 = 0.0
        loc_auc = 0.0

    # 计算评估时的分类错误率（使用与训练时相同的逻辑）
    # 训练时：通过criterion(outputs, targets)计算class_error
    # 验证时：我们也使用同样的criterion来计算class_error
    eval_class_error = 0.0

    # 只在非定位任务模式下计算分类错误率
    if not localization_only:
        # 收集所有批次的class_error
        total_class_error = 0.0
        total_batches = 0

        # 重新遍历数据来计算class_error
        model.eval()
        criterion.eval()

        for samples, targets in data_loader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            outputs = model(samples)
            loss_dict = criterion(outputs, targets)

            if 'class_error' in loss_dict:
                total_class_error += loss_dict['class_error'].item()
                total_batches += 1

        if total_batches > 0:
            eval_class_error = total_class_error / total_batches
        else:
            eval_class_error = 100.0

        print(f"[DEBUG] 使用训练时相同逻辑计算的分类错误率: {eval_class_error:.2f}%")
    else:
        print(f"[DEBUG] 定位任务模式：跳过分类错误率计算")

    # 将评估指标保存到JSON文件
    test_metrics = {
        'epoch': epoch,
        'mAP': mean_ap,
        'mPrecision': mean_precision,
        'mRecall': mean_recall,
        'mAP_F1': map_f1,
        'class_APs': class_aps,
        'sample_level': {
            'f1': sample_f1,
            'auc': sample_auc
        },
        'localization': {
            'f1': loc_f1,
            'auc': loc_auc
        },
        'correct_boxes': total_correct_boxes,
        'total_predicted_boxes': total_predicted_boxes,
        'total_target_boxes': total_target_boxes,
        'false_positive': total_false_positive,
        'false_negative': total_false_negative,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'loss': metric_logger.meters['loss'].global_avg,
        'debug_info': debug_info  # 添加DEBUG信息到JSON
    }

    # 只在非定位任务模式下添加分类指标
    if not localization_only:
        test_metrics['classification'] = {
            'f1': class_f1_macro,
            'auc': class_auc
        }
        test_metrics['eval_class_error'] = eval_class_error  # 使用评估时的分类错误率

    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if fold is not None:
        filename = f'val_metrics_fold_{fold}_epoch_{epoch}.json'
    else:
        filename = f'val_metrics_epoch_{epoch}.json'
    with open(output_dir / filename, 'w') as f:
        json.dump(test_metrics, f, indent=4, ensure_ascii=False)

    # 在验证循环结束后，只在rank 0进程打印一次统计信息
    print(f"\n验证集指标:")
    if localization_only:
        print(f"🎯 定位任务模式 - 只计算定位指标:")
        print(f"位置F1: {loc_f1:.4f}, 位置AUC: {loc_auc:.4f}")
        print(
            f"mAP: {mean_ap:.4f}, mP: {mean_precision:.4f}, mR: {mean_recall:.4f}, mAP-F1: {map_f1:.4f}")
    else:
        print(
            f"分类F1: {class_f1_macro:.4f}, 分类AUC: {class_auc:.4f} (基于IoU匹配框)")
        print(f"分类错误率: {eval_class_error:.2f}% (基于匹配框，与训练时相同)")
        print(f"位置F1: {loc_f1:.4f}, 位置AUC: {loc_auc:.4f}")
        print(
            f"mAP: {mean_ap:.4f}, mP: {mean_precision:.4f}, mR: {mean_recall:.4f}, mAP-F1: {map_f1:.4f}")
    print(f"\n详细统计:")
    print(f"正确预测框数: {total_correct_boxes}")
    print(f"总预测框数: {total_predicted_boxes}")
    print(f"总目标框数: {total_target_boxes}")
    print(f"假阳性数量: {total_false_positive}")
    print(f"假阴性数量: {total_false_negative}")
    print(f"精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
    print(f"位置数据统计: 真实标签数={len(all_true_local)}, 预测置信度数={len(all_pred_local)}")
    if len(all_true_local) > 0:
        print(
            f"位置真实标签分布: 0={all_true_local.count(0)}, 1={all_true_local.count(1)}")
        print(
            f"位置预测置信度范围: [{min(all_pred_local):.4f}, {max(all_pred_local):.4f}]")
        print(f"CDS级别位置评估: 总CDS数={len(all_true_local)}")
        print(f"  - 真实BGC区域CDS数: {all_true_local.count(1)}")
        print(f"  - 非BGC区域CDS数: {all_true_local.count(0)}")
        print(
            f"  - BGC区域覆盖率: {all_true_local.count(1)/len(all_true_local)*100:.2f}%")
    print("Averaged stats:", metric_logger)

    # 🔥 新增：CDS级别指标
    cds_level_metrics = {
        'precision': cds_precision if 'cds_precision' in locals() else 0,
        'recall': cds_recall if 'cds_recall' in locals() else 0,
        'f1_score': loc_f1,
        'total_cds_positions': len(all_true_local) if len(all_true_local) > 0 else 0,
        'true_positive': cds_tp if 'cds_tp' in locals() else 0,
        'false_positive': cds_fp if 'cds_fp' in locals() else 0,
        'false_negative': cds_fn if 'cds_fn' in locals() else 0
    }

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats.update({
        'mAP': mean_ap,
        'mPrecision': mean_precision,
        'mRecall': mean_recall,
        'mAP_F1': map_f1,
        'sample_level_f1': sample_f1,
        'sample_level_auc': sample_auc,
        'localization_f1': loc_f1,
        'localization_auc': loc_auc,
        'correct_boxes': total_correct_boxes,
        'total_predicted_boxes': total_predicted_boxes,
        'total_target_boxes': total_target_boxes,
        'false_positive': total_false_positive,
        'false_negative': total_false_negative,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cds_level_metrics': cds_level_metrics  # 🔥 新增CDS级别指标
    })

    # 只在非定位任务模式下添加分类指标
    if not localization_only:
        stats.update({
            'classification_f1': class_f1_macro,
            'classification_auc': class_auc,
        })
    return stats


# 分布式标签聚合函数
def gather_all_labels(local_labels, device):
    """
    分布式标签聚合函数

    Args:
        local_labels: 本地标签列表
        device: 计算设备

    Returns:
        list: 全局标签列表
    """
    import torch
    local_tensor = torch.tensor(local_labels, device=device, dtype=torch.int32)
    local_len = torch.tensor([len(local_labels)], device=device)
    world_size = torch.distributed.get_world_size(
    ) if torch.distributed.is_initialized() else 1
    if world_size == 1:
        return local_labels
    all_lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    torch.distributed.all_gather(all_lens, local_len)
    max_len = max([l.item() for l in all_lens])
    if len(local_labels) < max_len:
        local_tensor = torch.cat([local_tensor, torch.zeros(
            max_len - len(local_labels), dtype=torch.int32, device=device)])
    all_labels = [torch.zeros(max_len, dtype=torch.int32, device=device)
                  for _ in range(world_size)]
    torch.distributed.all_gather(all_labels, local_tensor)
    result = []
    for t, l in zip(all_labels, all_lens):
        result.extend(t[:l].tolist())
    return result


# 新增：分布式概率分布聚合函数


def gather_all_probs(local_probs, device, num_classes):
    """
    分布式概率分布聚合函数

    Args:
        local_probs: 本地概率分布列表
        device: 计算设备
        num_classes: 类别数量

    Returns:
        list: 全局概率分布列表
    """
    import torch
    import numpy as np
    local_probs = np.array(local_probs, dtype=np.float32)
    if local_probs.ndim == 1:
        # 只有一个样本时 shape 可能是 (num_classes,)
        local_probs = local_probs[None, :]
    if local_probs.shape[0] == 0:
        # 没有样本时 shape 应为 (0, num_classes)
        local_probs = np.zeros((0, num_classes), dtype=np.float32)
    local_tensor = torch.tensor(
        local_probs, device=device, dtype=torch.float32)
    local_len = torch.tensor([local_tensor.shape[0]], device=device)
    world_size = torch.distributed.get_world_size(
    ) if torch.distributed.is_initialized() else 1
    if world_size == 1:
        return local_probs.tolist()
    all_lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    torch.distributed.all_gather(all_lens, local_len)
    max_len = max([l.item() for l in all_lens])
    # 补零到最大长度
    if local_tensor.shape[0] < max_len:
        pad = torch.zeros(
            (max_len - local_tensor.shape[0], num_classes), dtype=torch.float32, device=device)
        local_tensor = torch.cat([local_tensor, pad], dim=0)
    all_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    torch.distributed.all_gather(all_tensors, local_tensor)
    # 截断补零部分
    all_probs = []
    for t, l in zip(all_tensors, all_lens):
        all_probs.append(t[:l.item()].cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    return all_probs.tolist()
