#!/usr/bin/env python3
"""Simple test without full TimeEval framework"""

import numpy as np
import pandas as pd

# 簡単な異常検知アルゴリズム
def simple_anomaly_detection(data, threshold=2):
    """Simple anomaly detection using z-score"""
    if isinstance(data, pd.DataFrame):
        # 各列のz-scoreを計算
        z_scores = np.abs((data - data.mean()) / data.std())
        # 各行の最大z-scoreを異常スコアとする
        scores = z_scores.max(axis=1)
    else:
        # 1次元の場合
        z_scores = np.abs((data - data.mean()) / data.std())
        scores = z_scores
    
    # 0-1の範囲に正規化
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores.fillna(0) if isinstance(scores, pd.Series) else scores

# テストデータ生成
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 5))
anomaly_indices = [100, 200, 300, 400, 500]
normal_data[anomaly_indices] += 5  # 異常値を追加

# テスト実行
df = pd.DataFrame(normal_data, columns=[f'channel_{i}' for i in range(5)])
scores = simple_anomaly_detection(df)

print("Test completed successfully!")
print(f"Data shape: {df.shape}")
print(f"Anomaly scores shape: {scores.shape}")
print(f"Score range: {scores.min():.3f} - {scores.max():.3f}")
print(f"Top 10 anomaly indices: {np.argsort(scores)[-10:]}")