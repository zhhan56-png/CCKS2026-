#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCKS2026 大模型生成文本检测及溯源 —— 本地验证脚本
=================================================

用 80/20 训练/验证划分估算线上得分：
  Final = 0.8 * F1_detect + 0.2 * F1_source

划分使用分层采样（stratify=labels），保证验证集类别比例与训练集一致。

用法示例：
  python evaluate.py --train train.jsonl --test-size 0.2 --seed 42
"""

import argparse
import json
import time
import warnings

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

FAMILY_NAMES = {
    0: 'OpenAI', 1: 'Alibaba', 2: 'DeepSeek', 3: 'ByteDance',
    4: 'Moonshot', 5: 'Google', 6: 'Anthropic', 7: 'xAI',
}


def load_jsonl(path):
    """读取 JSONL 文件，返回记录列表。"""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser(description='CCKS2026 AI 文本检测与溯源 - 本地验证')
    parser.add_argument('--train', default='train.jsonl', help='训练数据路径')
    parser.add_argument('--test-size', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    print("=" * 60)
    print("EVALUATION (80/20 split, word + char features)")
    print("=" * 60)

    raw = load_jsonl(args.train)

    # Prepare arrays
    ids = np.arange(len(raw))
    texts = np.array([d['text'] for d in raw])
    labels = np.array([d['label'] for d in raw])
    families = np.array([d.get('family', -1) for d in raw])

    # Split (stratified by label)
    tr_idx, val_idx = train_test_split(
        ids, test_size=args.test_size, random_state=args.seed, stratify=labels
    )

    train_texts = texts[tr_idx]
    val_texts = texts[val_idx]
    y_train = labels[tr_idx]
    y_val = labels[val_idx]
    fam_train = families[tr_idx]
    fam_val = families[val_idx]

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    # ===== DETECTION =====
    print("\n--- Detection (3-class) ---")
    t0 = time.time()

    vchar = TfidfVectorizer(
        max_features=25000, analyzer='char_wb', ngram_range=(2, 5),
        sublinear_tf=True, min_df=3, max_df=0.8, dtype=np.float32,
    )
    vword = TfidfVectorizer(
        max_features=10000, analyzer='word', ngram_range=(1, 2),
        sublinear_tf=True, min_df=5, max_df=0.8, dtype=np.float32,
    )
    x_tr = hstack([vchar.fit_transform(train_texts.tolist()),
                   vword.fit_transform(train_texts.tolist())]).tocsr()
    x_val = hstack([vchar.transform(val_texts.tolist()),
                    vword.transform(val_texts.tolist())]).tocsr()
    print(f"  Features built ({time.time()-t0:.1f}s)")

    lr = LogisticRegression(C=3.0, solver='saga', max_iter=200, tol=1e-4, random_state=42)
    lr.fit(x_tr, y_train)
    y_pred = lr.predict(x_val)

    f1_detect = f1_score(y_val, y_pred, average='macro')
    dp = f1_score(y_val, y_pred, average=None)
    print(f"  Macro-F1: {f1_detect:.4f}")
    print(f"  Class 0={dp[0]:.4f}, 1={dp[1]:.4f}, 2={dp[2]:.4f}")

    # ===== SOURCE TRACING =====
    print("\n--- Source Tracing (8-class) ---")

    # Find true machine samples in val
    val_machine_mask = (y_val == 1)
    val_machine_texts = val_texts[val_machine_mask]
    val_machine_families = fam_val[val_machine_mask]
    val_machine_preds = y_pred[val_machine_mask]

    print(f"  True machine in val: {len(val_machine_texts)}")
    print(f"  Predicted as machine: {np.sum(val_machine_preds == 1)}")

    if len(val_machine_texts) > 0:
        # Train source model on training machine samples
        tr_machine_mask = (y_train == 1)
        tr_machine_texts = train_texts[tr_machine_mask]
        tr_machine_families = fam_train[tr_machine_mask]

        t0 = time.time()
        sc = TfidfVectorizer(max_features=15000, analyzer='char_wb', ngram_range=(2, 5),
                             sublinear_tf=True, min_df=3, max_df=0.9, dtype=np.float32)
        sw = TfidfVectorizer(max_features=8000, analyzer='word', ngram_range=(1, 2),
                             sublinear_tf=True, min_df=3, max_df=0.9, dtype=np.float32)
        x_st_tr = hstack([sc.fit_transform(tr_machine_texts.tolist()),
                          sw.fit_transform(tr_machine_texts.tolist())]).tocsr()

        lr_src = LogisticRegression(C=3.0, solver='saga', max_iter=200, tol=1e-4, random_state=42)
        lr_src.fit(x_st_tr, tr_machine_families)
        print(f"  Source model trained ({time.time()-t0:.1f}s)")

        # Evaluate
        true_fams = val_machine_families.tolist()
        pred_fams = []
        for i in range(len(val_machine_texts)):
            if val_machine_preds[i] == 1:
                f = hstack([sc.transform([val_machine_texts[i]]),
                            sw.transform([val_machine_texts[i]])])
                pred_fams.append(int(lr_src.predict(f)[0]))
            else:
                pred_fams.append(-1)

        valid = [(t, p) for t, p in zip(true_fams, pred_fams) if 0 <= p <= 7]
        if valid:
            vt = [v[0] for v in valid]
            vp = [v[1] for v in valid]
            f1_source = f1_score(vt, vp, average='macro')
            sp = f1_score(vt, vp, average=None)
            print(f"  Valid for source eval: {len(valid)}/{len(true_fams)}")
            print(f"  Macro-F1: {f1_source:.4f}")
            for i in range(8):
                print(f"    {FAMILY_NAMES[i]} ({i}): {sp[i]:.4f}")
        else:
            f1_source = 0.0
            print("  Source F1: 0")
    else:
        f1_source = 0.0

    final = 0.8 * f1_detect + 0.2 * f1_source
    print(f"\n{'=' * 60}")
    print(f"FINAL ESTIMATED SCORE")
    print(f"{'=' * 60}")
    print(f"F1_detect: {f1_detect:.6f}")
    print(f"F1_source: {f1_source:.6f}")
    print(f"Final:     {final:.6f}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
