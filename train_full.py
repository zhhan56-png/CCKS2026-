#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCKS2026 大模型生成文本检测及溯源 —— 主训练与提交脚本
=====================================================

方案：字符级 + 单词级 TF-IDF 特征，双阶段 Logistic Regression (saga solver)

流程：
  1. 加载训练数据（三分类 label：0=人类撰写 / 1=纯机器生成 / 2=人机协作生成）
  2. 检测任务：char(2-5元) + word(1-2元) TF-IDF 拼接 → LR 三分类
  3. 溯源任务：仅用 label==1 的样本，同样特征 → LR 八分类（8 个模型家族）
  4. 预测测试集：label==1 的样本输出溯源结果，其余 family=-1
  5. 生成 submit.jsonl，并保存 pipeline.pkl（向量器 + 模型）

用法示例：
  python train_full.py \
      --train train.jsonl \
      --test test_a_release.jsonl \
      --output submit.jsonl \
      --model pipeline.pkl
"""

import argparse
import json
import pickle
import time
import warnings
from collections import Counter

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')


# 8 个模型家族（溯源任务类别）
FAMILY_NAMES = {
    0: 'OpenAI', 1: 'Alibaba', 2: 'DeepSeek', 3: 'ByteDance',
    4: 'Moonshot', 5: 'Google', 6: 'Anthropic', 7: 'xAI',
}


def load_jsonl(path):
    """读取 JSONL 文件，返回记录列表。"""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def build_detect_features(texts):
    """检测任务特征：字符 n-gram + 单词 n-gram 的 TF-IDF 拼接。"""
    vec_char = TfidfVectorizer(
        max_features=25000,
        analyzer='char_wb',
        ngram_range=(2, 5),
        sublinear_tf=True,
        min_df=3,
        max_df=0.8,
        dtype=np.float32,
    )
    vec_word = TfidfVectorizer(
        max_features=10000,
        analyzer='word',
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=5,
        max_df=0.8,
        dtype=np.float32,
    )
    x_char = vec_char.fit_transform(texts)
    x_word = vec_word.fit_transform(texts)
    x = hstack([x_char, x_word]).tocsr()
    return x, (vec_char, vec_word)


def build_source_features(texts):
    """溯源任务特征：参数与检测任务略作调整，以捕捉模型风格细节。"""
    vec_char = TfidfVectorizer(
        max_features=15000,
        analyzer='char_wb',
        ngram_range=(2, 5),
        sublinear_tf=True,
        min_df=3,
        max_df=0.9,
        dtype=np.float32,
    )
    vec_word = TfidfVectorizer(
        max_features=8000,
        analyzer='word',
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.9,
        dtype=np.float32,
    )
    x_char = vec_char.fit_transform(texts)
    x_word = vec_word.fit_transform(texts)
    x = hstack([x_char, x_word]).tocsr()
    return x, (vec_char, vec_word)


def train_detection(x, labels):
    """训练三分类检测模型，并打印训练集 Macro-F1。"""
    model = LogisticRegression(
        C=3.0, solver='saga', max_iter=200, tol=1e-4, random_state=42
    )
    model.fit(x, labels)
    train_pred = model.predict(x)
    train_f1 = f1_score(labels, train_pred, average='macro')
    per_class = f1_score(labels, train_pred, average=None)
    print(f"  Training F1: {train_f1:.4f}")
    print(f"  Per-class: 0={per_class[0]:.4f}, 1={per_class[1]:.4f}, 2={per_class[2]:.4f}")
    print(f"  Iterations: {model.n_iter_}")
    return model


def train_source(x, families):
    """训练八分类溯源模型（仅机器生成样本）。"""
    model = LogisticRegression(
        C=3.0, solver='saga', max_iter=200, tol=1e-4, random_state=42
    )
    model.fit(x, families)
    src_pred = model.predict(x)
    src_f1 = f1_score(families, src_pred, average='macro')
    print(f"  Training F1: {src_f1:.4f}")
    print(f"  Iterations: {model.n_iter_}")
    return model


def save_submission(test_ids, detect_preds, source_preds, output_path):
    """组装提交文件：仅 label==1 的样本写入溯源 family，其余为 -1。"""
    results = []
    for tid, lbl, family in zip(test_ids, detect_preds, source_preds):
        lbl = int(lbl)
        results.append({
            'id': tid,
            'label': lbl,
            'family': int(family) if lbl == 1 else -1,
        })

    label_dist = Counter(r['label'] for r in results)
    print(f"  Label distribution: {dict(label_dist)}")
    print(f"  Source tracing count: {sum(1 for r in results if r['label'] == 1)}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"  Saved {output_path} [OK]")


def main():
    parser = argparse.ArgumentParser(description='CCKS2026 AI 文本检测与溯源 - 训练与提交')
    parser.add_argument('--train', default='train.jsonl', help='训练数据路径')
    parser.add_argument('--test', default='test_a_release.jsonl', help='测试数据路径')
    parser.add_argument('--output', default='submit.jsonl', help='提交文件输出路径')
    parser.add_argument('--model', default='pipeline.pkl', help='模型保存路径')
    args = parser.parse_args()

    t_start = time.time()

    # ============================================================
    # 1. Load Data
    # ============================================================
    print("=" * 60)
    print("AI-Generated Text Detection & Source Tracing")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    train_data = load_jsonl(args.train)
    texts = [d['text'] for d in train_data]
    labels = np.array([d['label'] for d in train_data])
    print(f"  Training: {len(texts)} samples")
    print(f"  Labels: 0={sum(labels==0)}, 1={sum(labels==1)}, 2={sum(labels==2)}")

    # ============================================================
    # 2. Detection Features (char n-gram TF-IDF)
    # ============================================================
    print("\n[2/5] Building detection features...")
    t0 = time.time()
    x_detect, (vec_detect_char, vec_detect_word) = build_detect_features(texts)
    print(f"  Shape: {x_detect.shape} ({time.time()-t0:.1f}s)")

    # ============================================================
    # 3. Train Detection Model
    # ============================================================
    print("\n[3/5] Training detection model (LR + saga)...")
    t0 = time.time()
    detect_model = train_detection(x_detect, labels)
    print(f"  ({time.time()-t0:.1f}s)")

    # ============================================================
    # 4. Source Tracing Model
    # ============================================================
    print("\n[4/5] Training source tracing model...")
    machine_data = [d for d in train_data if d['label'] == 1]
    machine_texts = [d['text'] for d in machine_data]
    machine_families = np.array([d['family'] for d in machine_data])
    print(f"  Machine samples: {len(machine_texts)}")

    t0 = time.time()
    x_source, (vec_source_char, vec_source_word) = build_source_features(machine_texts)
    print(f"  Source features shape: {x_source.shape} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    source_model = train_source(x_source, machine_families)
    print(f"  ({time.time()-t0:.1f}s)")

    # ============================================================
    # 5. Predict Test A & Generate Submission
    # ============================================================
    print("\n[5/5] Predicting on test_a...")
    test_data = load_jsonl(args.test)
    test_texts = [d['text'] for d in test_data]
    test_ids = [d['id'] for d in test_data]
    print(f"  Test samples: {len(test_texts)}")

    # Detection features
    x_test_char = vec_detect_char.transform(test_texts)
    x_test_word = vec_detect_word.transform(test_texts)
    x_test_detect = hstack([x_test_char, x_test_word]).tocsr()
    detect_preds = detect_model.predict(x_test_detect)

    # Source features
    x_test_src_char = vec_source_char.transform(test_texts)
    x_test_src_word = vec_source_word.transform(test_texts)
    x_test_source = hstack([x_test_src_char, x_test_src_word]).tocsr()
    source_preds = source_model.predict(x_test_source)

    save_submission(test_ids, detect_preds, source_preds, args.output)

    # Save models
    artifacts = {
        'detect_model': detect_model,
        'source_model': source_model,
        'vec_detect_char': vec_detect_char,
        'vec_detect_word': vec_detect_word,
        'vec_source_char': vec_source_char,
        'vec_source_word': vec_source_word,
    }
    with open(args.model, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"  Saved {args.model} [OK]")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"DONE! Total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
