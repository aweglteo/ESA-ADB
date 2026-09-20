#!/usr/bin/env python3
"""
ChannelAwareFScore metrics calculator for research_experiments results.
Copied logic from timeeval.metrics.ranking_metrics.ChannelAwareFScore
"""

import argparse
import json
import pandas as pd
import numpy as np
import portion as P
from pathlib import Path
from typing import Dict, Optional


def convert_time_series_to_events(vector):
    """
    Convert time series to events intervals.
    Copied from timeeval.metrics.utils.convert_time_series_to_events
    """
    vector = np.asarray(vector)

    def find_runs(x):
        """Find runs of consecutive items in an array."""
        x = np.asanyarray(x)
        if x.ndim != 1:
            raise ValueError('only 1D array supported')
        n = x.shape[0]

        if n == 0:
            return np.array([]), np.array([]), np.array([])
        else:
            loc_run_start = np.empty(n, dtype=bool)
            loc_run_start[0] = True
            np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
            run_starts = np.nonzero(loc_run_start)[0]
            run_values = x[loc_run_start]
            run_lengths = np.diff(np.append(run_starts, n))
            run_ends = run_starts + run_lengths
            
            return run_values, run_starts, run_ends

    timestamps, values = vector[:, 0], vector[:, 1]

    values, starts, ends = find_runs(values)
    events = []

    for i, val in enumerate(values):
        if val == 1:
            start_time = timestamps[starts[i]]
            end_time = timestamps[ends[i] - 1] if ends[i] <= len(timestamps) else timestamps[-1]
            events.append(P.closed(start_time, end_time))

    return P.Interval(*events)


class ChannelAwareFScore:
    """
    ChannelAwareFScore implementation copied from timeeval.metrics.ranking_metrics
    """
    
    def __init__(self, beta: float = 0.5, select_labels: Optional[dict] = None, 
                 full_range: Optional[tuple] = None, name: Optional[str] = None):
        self._beta = beta
        self.full_range = full_range
        
        if select_labels is None or len(select_labels) == 0:
            self.selected_labels = dict()
            filter_string = "ALL"
        else:
            select_labels = {col: np.atleast_1d(val) for col, val in select_labels.items()}
            self.selected_labels = select_labels
            filter_string = "_".join(["_".join(val) for val in select_labels.values()])
        self._name = f"PC_{filter_string}" if name is None else name

    def get_pr_re_f_score(self, true_positives, false_positives, false_negatives):
        """Calculate precision, recall, F-score"""
        divider = true_positives + false_positives
        if divider == 0:
            precision = 0.0
        else:
            precision = true_positives / divider

        divider = true_positives + false_negatives
        if divider == 0:
            recall = 0.0
        else:
            recall = true_positives / divider

        divider = (self._beta ** 2) * precision + recall
        if divider == 0:
            f_score = 0.0
        else:
            f_score = ((1 + self._beta ** 2) * precision * recall) / divider

        return precision, recall, f_score

    def score(self, y_true: pd.DataFrame, y_pred: dict, subsystems_mapping: dict = None) -> Dict[str, float]:
        """
        Calculate ChannelAwareFScore - logic copied from original implementation
        """
        all_channels = list(y_pred.keys())

        for c, preds in y_pred.items():
            y_pred[c] = np.asarray(preds)

        # Set full range
        min_y_pred_timestamp = min([preds[..., 0].min() for preds in y_pred.values()])
        max_y_pred_timestamp = max([preds[..., 0].max() for preds in y_pred.values()])
        
        if self.full_range is None:
            self.full_range = (min(y_true["StartTime"].min(), min_y_pred_timestamp), 
                              max(y_true["EndTime"].max(), max_y_pred_timestamp))
        
        # Extend predictions to full range
        for c, preds in y_pred.items():
            if y_pred[c][0, 0] > self.full_range[0]:
                y_pred[c] = np.array([np.array([self.full_range[0], y_pred[c][0, 1]]), *y_pred[c]])
            if y_pred[c][-1, 0] < self.full_range[1]:
                y_pred[c] = np.array([*y_pred[c], np.array([self.full_range[1], y_pred[c][-1, 1]])])

        # Convert predictions to events
        events_pred_per_channel = dict()
        for channel_name, channel_pred in y_pred.items():
            events_pred_per_channel[channel_name] = convert_time_series_to_events(np.asarray(channel_pred))

        # Filter ground truth labels
        filtered_y_true = y_true.copy()
        for col, val in self.selected_labels.items():
            filtered_y_true = filtered_y_true[filtered_y_true[col].isin(val)]

        # Fix for point anomalies
        point_anomalies = (filtered_y_true["StartTime"] == filtered_y_true["EndTime"])
        filtered_y_true.loc[point_anomalies, "EndTime"] = filtered_y_true.loc[point_anomalies, "StartTime"] + pd.Timedelta(milliseconds=1)

        # Process each unique anomaly ID
        unique_ids = filtered_y_true["ID"].unique()
        global_precisions = []
        global_recalls = []
        global_f_scores = []
        
        # Build intervals for each anomaly ID and channel
        aid_channels_intervals = dict()
        for aid in unique_ids:
            gt = filtered_y_true[filtered_y_true["ID"] == aid]
            
            channels_intervals = dict()
            for c in all_channels:
                c_gt = gt[gt["Channel"] == c]
                c_gt_intervals = []
                for _, row in c_gt[["StartTime", "EndTime"]].iterrows():
                    c_gt_intervals.append(P.closed(*row))
                channels_intervals[c] = P.Interval(*c_gt_intervals)
            aid_channels_intervals[aid] = channels_intervals

        # Calculate metrics for each anomaly ID
        for aid in unique_ids:
            channels_intervals = aid_channels_intervals[aid]

            # Create full interval for this anomaly
            full_interval = []
            for interval in channels_intervals.values():
                full_interval.append(interval)
            full_interval = P.Interval(*full_interval)

            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            for i, c in enumerate(all_channels):
                is_channel_affected = not channels_intervals[c].empty
                detection_interval = full_interval & events_pred_per_channel[c]
                is_channel_detected = not detection_interval.empty
                
                if is_channel_affected and is_channel_detected:
                    true_positives += 1
                elif is_channel_affected and not is_channel_detected:
                    false_negatives += 1
                elif not is_channel_affected and is_channel_detected:
                    # Remove any false detections that overlap with true positives for other anomalies
                    for id, a_intervals in aid_channels_intervals.items():
                        if aid == id:
                            continue
                        if a_intervals[c].empty:
                            continue
                        if not (detection_interval & a_intervals[c]).empty:
                            break
                    else:
                        false_positives += 1

            precision, recall, f_score = self.get_pr_re_f_score(true_positives, false_positives, false_negatives)

            global_precisions.append(precision)
            global_recalls.append(recall)
            global_f_scores.append(f_score)

        print(f"Individual F-scores: {global_f_scores}")

        result_dict = {
            "channel_precision": np.mean(global_precisions),
            "channel_recall": np.mean(global_recalls),
            f"channel_F{self._beta:.2f}": np.mean(global_f_scores)
        }

        return result_dict

    @property
    def name(self) -> str:
        return self._name


def load_ground_truth_labels():
    """Ground truthラベルを読み込む"""
    labels_path = Path("data/ESA-Mission1/labels.csv")
    labels_df = pd.read_csv(labels_path)
    
    # StartTime, EndTimeをdatetimeに変換（UTCタイムゾーンを除去）
    labels_df['StartTime'] = pd.to_datetime(labels_df['StartTime']).dt.tz_localize(None)
    labels_df['EndTime'] = pd.to_datetime(labels_df['EndTime']).dt.tz_localize(None)
    
    return labels_df


def load_predictions(experiment_dir, channel_name):
    """チャンネルの予測結果を読み込み、ChannelAwareFScore用の形式に変換"""
    # labels/以下のparquetファイルを使用
    pred_file = Path(experiment_dir) / "labels" / f"{channel_name}.parquet"
    
    if not pred_file.exists():
        print(f"Prediction file not found: {pred_file}")
        return None
    
    df = pd.read_parquet(pred_file)
    print(f"  Loaded {channel_name} parquet - shape: {df.shape}, columns: {df.columns.tolist()}")
    
    # カラム名に応じて処理
    if 'timestamp' in df.columns and 'label' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'])
        is_anomaly = df['label'].astype(int)
    elif 'timestamp' in df.columns and 'is_anomaly' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'])
        is_anomaly = df['is_anomaly'].astype(int)
    else:
        # カラム名を確認して適切に処理
        print(f"  Available columns: {df.columns.tolist()}")
        print(f"  Sample data:\n{df.head()}")
        return None
    
    # numpy arrayの形式に変換: [[timestamp, is_anomaly], ...]
    result = np.empty((len(timestamps), 2), dtype=object)
    result[:, 0] = timestamps
    result[:, 1] = is_anomaly
    
    return result


def load_all_predictions(experiment_dir):
    """すべてのチャンネルの予測結果を読み込む"""
    exp_path = Path(experiment_dir)
    y_pred = {}
    
    for channel_dir in exp_path.iterdir():
        if channel_dir.is_dir() and channel_dir.name.startswith("channel_"):
            channel_name = channel_dir.name
            predictions = load_predictions(experiment_dir, channel_name)
            
            if predictions is not None:
                y_pred[channel_name] = predictions
                print(f"Loaded {channel_name}: {predictions.shape[0]} time points")
            else:
                print(f"Failed to load {channel_name}")
    
    return y_pred


def calculate_channel_aware_fscore(experiment_dir, beta=0.5, single_channel=None):
    """ChannelAwareFScoreを計算"""
    # データ読み込み
    y_true = load_ground_truth_labels()
    
    if single_channel:
        # 単一チャンネルのみテスト
        print(f"\nTesting single channel: {single_channel}")
        y_pred = {single_channel: load_predictions(experiment_dir, single_channel)}
        target_channels = [single_channel]
    else:
        y_pred = load_all_predictions(experiment_dir)
        target_channels = list(y_pred.keys())
    
    # 対象チャンネルのみフィルタ
    y_true_filtered = y_true[y_true["Channel"].isin(target_channels)]
    
    # 時間範囲の確認と調整
    if y_pred:
        first_channel = list(y_pred.keys())[0]
        pred_data = y_pred[first_channel]
        pred_start = pred_data[0, 0]
        pred_end = pred_data[-1, 0]
        
        print(f"Prediction time range: {pred_start} to {pred_end}")
        print(f"Ground truth time range (before filter): {y_true_filtered['StartTime'].min()} to {y_true_filtered['EndTime'].max()}")
        
        # Ground truthを予測期間でフィルタ
        # 異常期間が予測期間とオーバーラップするもののみ残す
        mask = (y_true_filtered['EndTime'] >= pred_start) & (y_true_filtered['StartTime'] <= pred_end)
        y_true_filtered = y_true_filtered[mask]
        
        print(f"Ground truth time range (after filter): {y_true_filtered['StartTime'].min()} to {y_true_filtered['EndTime'].max()}")
    
    print(f"\nFiltered labels: {y_true_filtered.shape[0]} records")
    print(f"Target channels: {target_channels}")
    
    # デバッグ情報を追加
    print(f"Unique anomaly IDs: {len(y_true_filtered['ID'].unique())}")
    print(f"Sample IDs: {y_true_filtered['ID'].unique()[:10]}")
    
    # ChannelAwareFScoreを計算
    metric = ChannelAwareFScore(beta=beta)
    result = metric.score(y_true_filtered, y_pred)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Calculate ChannelAwareFScore metrics")
    parser.add_argument("experiment_path", type=str, 
                       help="Path to experiment directory (e.g., research_experiments/20251125_163847)")
    parser.add_argument("--beta", type=float, default=0.5,
                       help="Beta parameter for F-beta score (default: 0.5)")
    parser.add_argument("--output", type=str, default="channel_aware_results.json",
                       help="Output file name (default: channel_aware_results.json)")
    parser.add_argument("--channel", type=str, default=None,
                       help="Test single channel only (e.g., channel_41)")
    
    args = parser.parse_args()
    
    try:
        print(f"Calculating ChannelAwareFScore for {args.experiment_path}...")
        result = calculate_channel_aware_fscore(args.experiment_path, beta=args.beta, single_channel=args.channel)
        
        print(f"\n=== ChannelAwareFScore Results (beta={args.beta}) ===")
        for key, value in result.items():
            print(f"{key}: {value:.4f}")
        
        # Save results to JSON
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())