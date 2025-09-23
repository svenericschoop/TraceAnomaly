"""
Data preprocessing module for trace analysis and anomaly detection.

This module contains all the data preprocessing functions used for:
- Trace processing and subtype aggregation
- Resource data (CPU/RAM) processing
- DeepLog sequence generation
- LSTM data preparation
- Utility functions for data manipulation
"""

import re
import json
import gzip
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence


# Constants
TRACE_START = "<TRACE_START>"
TRACE_END = "<TRACE_END>"


def aggregate_subtype_durations_per_trace(json_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregates subtype durations from traces.
    
    Args:
        json_path: Path to the compressed JSONL file containing traces
        
    Returns:
        Tuple of (DataFrame with aggregated subtype features, list of all subtypes found)
    """
    all_subtypes = set()
    trace_results = []

    def collect_durations(obj, durations):
        """Recursively collects subtype durations."""
        subtype = obj.get('subtype')
        duration = obj.get('duration')
        if subtype and duration and not pd.isna(subtype):
            durations[subtype].append(duration)
            all_subtypes.add(subtype)
        for child in obj.get('children', []):
            collect_durations(child, durations)

    with gzip.open(json_path, 'rt') as f:
        for line in f:
            trace = json.loads(line)
            durations = defaultdict(list)
            collect_durations(trace, durations)

            row = {'timestamp': trace.get('timestamp')}
            children = trace.get('children', [])
            row['duration'] = children[0].get('duration', 0) if children else 0

            for subtype, durs in durations.items():
                arr = np.array(durs)
                row.update({
                    f"{subtype}_avg": arr.mean(),
                    f"{subtype}_std": arr.std(),
                    f"{subtype}_min": arr.min(),
                    f"{subtype}_max": arr.max(),
                    f"{subtype}_count": len(arr)
                })
            trace_results.append(row)

    stat_names = ["avg", "std", "min", "max", "count"]
    all_columns = ["timestamp", "duration"] + [
        f"{subtype}_{stat}" for subtype in sorted(all_subtypes) for stat in stat_names
    ]
    df = pd.DataFrame(trace_results)
    df = df.reindex(columns=all_columns)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df.fillna(0, inplace=True)
    return df, sorted(all_subtypes)


def flatten_trace(trace_row: dict) -> List[str]:
    """
    Flattens a hierarchical trace into a list of spans sorted by timestamp.
    
    Args:
        trace_row: Dictionary containing trace data with children
        
    Returns:
        List of span names sorted by timestamp
    """
    spans = []

    def _flatten(span):
        spans.append({'name': span['name'], 'timestamp': span['timestamp']})
        for child in sorted(span.get('children', []), key=lambda x: x['timestamp']):
            _flatten(child)

    children = trace_row.get('children', [])
    if not children:
        return []
    _flatten(children[0])
    return [s['name'] for s in sorted(spans, key=lambda x: x['timestamp']) if s['name']]


def parse_span_names(span_name_sequences: List[List[str]], template_miner: TemplateMiner, 
                    regex_patterns: List[Tuple[str, str]], training_mode: bool = False) -> List[List]:
    """
    Parses span names using drain3 template miner to normalize and deduplicate patterns.
    
    Args:
        span_name_sequences: List of sequences of span names
        template_miner: Drain3 TemplateMiner instance
        regex_patterns: List of (pattern, replacement) tuples for regex normalization
        training_mode: If True, uses add_log_message() to build templates. If False, uses match() for inference.
        
    Returns:
        List of sequences with parsed span names
    """
    def normalize_span(span: str) -> str:
        for pattern, replacement in regex_patterns:
            span = re.sub(pattern, replacement, span)
        return span
    
    normalized = [
        [normalize_span(span) for span in seq]
        for seq in span_name_sequences
    ]
    all_span_names = [span for seq in normalized for span in seq]
    span_to_template = {}
    for span in all_span_names:
        if training_mode:
            # During training, use add_log_message to build template patterns
            result = template_miner.add_log_message(span)
            span_to_template[span] = result["cluster_id"] if result else 0
        else:
            # During testing, use match to find existing template patterns
            result = template_miner.match(span)
            span_to_template[span] = result.cluster_id if result else 0
    return [[span_to_template.get(span, span) for span in seq] for seq in normalized]


def generate_sequences_within_trace(events: List, seq_len: int) -> Tuple[List, List]:
    """
    Generates sequences and labels from a trace for deeplog.
    
    Args:
        events: List of events in the trace
        seq_len: Length of sequences to generate
        
    Returns:
        Tuple of (sequences, labels)
    """
    if len(events) <= seq_len:
        return [], []
    sequences = [events[i:i + seq_len] for i in range(len(events) - seq_len)]
    labels = [events[i + seq_len] for i in range(len(events) - seq_len)]
    return sequences, labels


def convert_traces_into_encoded_sequences(
        traces_file: str,
        drain_state_file: str,
        traces_df: pd.DataFrame,
        regex_patterns: List[Tuple[str, str]],
        seq_len: int = 10,
        add_trace_tokens: bool = True,
        debug: bool = False,
        training_mode: bool = False
) -> Tuple[np.ndarray, np.ndarray, int, pd.DataFrame]:
    """
    Converts traces into encoded sequences and labels for deeplog.
    
    Args:
        traces_file: Path to traces file
        drain_state_file: Path to drain3 state file
        traces_df: DataFrame with trace data
        regex_patterns: List of regex patterns for normalization
        seq_len: Length of sequences
        add_trace_tokens: Whether to add start/end tokens
        debug: Whether to print debug information
        training_mode: If True, uses add_log_message() to build templates. If False, uses match() for inference.
        
    Returns:
        Tuple of (X sequences, y labels, number of unique events, updated traces_df)
    """
    persistence = FilePersistence(drain_state_file)
    template_miner = TemplateMiner(persistence)

    log_interval = 10000
    sequence_counts = []
    X = []
    y = []

    with gzip.open(traces_file, "rt") as f:
        for idx, line in enumerate(f):
            if debug and idx % log_interval == 0:
                print(f"Processing trace {idx}")
            trace = json.loads(line)
            span_names = [[TRACE_START] + list(flatten_trace(trace)) + [TRACE_END]] if add_trace_tokens else [list(flatten_trace(trace))]
            parsed_span_names = parse_span_names(span_names, template_miner, regex_patterns, training_mode)[0]
            seqs, labs = generate_sequences_within_trace(parsed_span_names, seq_len)
            sequence_counts.append(len(seqs))
            X.extend(seqs)
            y.extend(labs)

    X = np.array(X)
    y = np.array(y)

    traces_df["num_sequences"] = sequence_counts
    traces_df["num_sequences"] = traces_df["num_sequences"].fillna(0).astype(int)

    num_events = len(set(y))

    return X, y, num_events, traces_df


def prepare_resource_data(
        data: pd.DataFrame,
        value_col: str = "val",
        resource_name: str = "cpu",
        time_offset: int = 0
) -> pd.DataFrame:
    """
    Prepares resource data (CPU or RAM) by exploding lists, converting timestamps,
    adjusting for time offset, aggregating by time, and extracting statistical features.
    
    Args:
        data: DataFrame with resource data
        value_col: Name of the value column
        resource_name: Name of the resource (cpu/ram)
        time_offset: Time offset in hours
        
    Returns:
        Processed DataFrame with statistical features
    """
    data = data[:-1]

    data = data.explode(["time", value_col])
    data["time"] = pd.to_datetime(data["time"], unit='ms')
    data["time"] = data["time"] + pd.Timedelta(hours=time_offset)

    data = data.groupby("time").agg({value_col: list}).reset_index()
    data = data.rename(columns={value_col: resource_name})

    data = extract_resource_features(data, col=resource_name)
    return data


def extract_resource_features(df: pd.DataFrame, col: str = "cpu") -> pd.DataFrame:
    """
    Extracts statistical features from resource data.
    
    Args:
        df: DataFrame with resource data
        col: Name of the resource column
        
    Returns:
        DataFrame with additional statistical feature columns
    """
    stats = {
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max,
        "count": lambda x: len(x)
    }
    for name, func in stats.items():
        df[f"{col}_{name}"] = df[col].apply(lambda x: func(x) if isinstance(x, list) and len(x) > 0 else 0)
    return df


def expand_resource_data_by_time(resource_data: pd.DataFrame, resource_col: str = "cpu", 
                                max_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Expands resource_data so that for each second between time n and n+1, all columns are duplicated.
    
    Args:
        resource_data: DataFrame with resource data
        resource_col: Name of the resource column
        max_time: Maximum timestamp to expand to
        
    Returns:
        Expanded DataFrame with a row for each second
    """
    expanded = []
    resource_times = resource_data["time"].sort_values().reset_index(drop=True)

    for idx in range(len(resource_times) - 1):
        start = resource_times[idx]
        end = resource_times[idx + 1]
        row = resource_data.loc[resource_data["time"] == start].iloc[0]

        expanded_row = {}
        for col in resource_data.columns:
            val = row[col]
            expanded_row[col] = val
        seconds = pd.date_range(start=start, end=end - pd.Timedelta(seconds=1), freq='1s')
        for ts in seconds:
            expanded.append({'timestamp': ts, **expanded_row})

    # Handle the last timestamp
    last_time = resource_times.iloc[-1]
    row = resource_data.loc[resource_data["time"] == last_time].iloc[0]
    expanded_row = {}
    for col in resource_data.columns:
        val = row[col]
        expanded_row[col] = val
    max_time = last_time if max_time is None else max_time
    seconds = pd.date_range(start=last_time, end=max_time, freq='1s')
    for ts in seconds:
        expanded.append({'timestamp': ts, **expanded_row})

    expanded = pd.DataFrame(expanded)
    expanded = expanded.drop(columns=["time"])
    expanded.drop(columns=resource_col, inplace=True)

    return expanded


def aggregate_durations(duration_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates durations to 1 second intervals, computing mean, std, min, max and
    event count per interval.
    
    Args:
        duration_df: DataFrame with duration data
        
    Returns:
        Aggregated DataFrame with statistical features and time features
    """
    duration_df.index = pd.to_datetime(duration_df.index, unit='us')
    duration_df['event_count'] = 0
    # Resample duration_df to 1 second intervals and aggregate with duration mean and count of events
    data_aggregated = duration_df.resample('1s').agg({
        'duration': ['mean', 'std', 'min', 'max'],
        'event_count': 'count'
    }).reset_index()
    data_aggregated.columns = ['timestamp', 'duration_mean', 'duration_std', 'duration_min', 'duration_max', 'event_count']
    data_aggregated['duration_mean'] = data_aggregated['duration_mean'].fillna(0)
    data_aggregated['duration_std'] = data_aggregated['duration_std'].fillna(0)
    data_aggregated['duration_min'] = data_aggregated['duration_min'].fillna(0)
    data_aggregated['duration_max'] = data_aggregated['duration_max'].fillna(0)

    return data_aggregated


def prepare_data_for_aggregated_lstm(
        traces_df: pd.DataFrame,
        cpu_data: pd.DataFrame,
        ram_data: pd.DataFrame,
        timesteps: int = 20,
        stride: int = 10,
        splits: Optional[int] = None,
        debug: bool = False
):
    """
    Combines CPU, RAM and duration of traces into a single df. Also adds time of day features (hour_sin, hour_cos). Scales data and transforms it for an lstm.
    
    Args:
        traces_df: DataFrame with trace data
        cpu_data: DataFrame with CPU data
        ram_data: DataFrame with RAM data
        timesteps: Number of timesteps for LSTM
        stride: Stride for sequence generation
        splits: Number of splits for time series cross-validation (optional)
        debug: Whether to print debug information
        
    Returns:
        If splits is None: Tuple of (X data, scaler, aggregated data)
        If splits is provided: Tuple of (X_train_sets, X_test_sets, scaler, aggregated data)
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    aggregated_data = aggregate_durations(traces_df)

    def merge_with_expanded_resource_data(aggregated_data, resource_data, resource_col):
        """
        Expands resource data by time and merges it with the aggregated data.
        """
        expanded_data = expand_resource_data_by_time(
            resource_data, resource_col=resource_col, max_time=aggregated_data['timestamp'].max()
        )
        merged_data = pd.merge(
            aggregated_data,
            expanded_data,
            on='timestamp',
            how='left'
        )
        return merged_data

    data_aggregated_with_cpu = merge_with_expanded_resource_data(aggregated_data, cpu_data, resource_col='cpu')
    data_aggregated_with_cpu_ram = merge_with_expanded_resource_data(data_aggregated_with_cpu, ram_data, resource_col='ram')

    data_aggregated_with_cpu_ram['hour_sin'] = np.sin(2 * np.pi * data_aggregated_with_cpu_ram['timestamp'].dt.hour / 24)
    data_aggregated_with_cpu_ram['hour_cos'] = np.cos(2 * np.pi * data_aggregated_with_cpu_ram['timestamp'].dt.hour / 24)

    if debug:
        print(f"Df transformed to time series with 1s intervals: event_count (per interval) and duration (mean, std, min, max per interval) and cpu (mean, std, min, max, count) and ram (mean, std, min, max, count) and time of day (hour_sin and hour_cos): {data_aggregated_with_cpu_ram.shape}")
        print(data_aggregated_with_cpu_ram.tail())

    scaler = RobustScaler()
    X = scaler.fit_transform(data_aggregated_with_cpu_ram.drop(columns=['timestamp']))

    # Reshape data for LSTM: (samples, timesteps, features)
    X = np.array([X[i:i+timesteps] for i in range(0, len(X) - timesteps, stride)])

    if splits is not None:
        # train test split
        X_train_sets = []
        X_test_sets = []

        for train_index, test_index in TimeSeriesSplit(n_splits=splits).split(X):
            X_train_sets.append(X[train_index])
            X_test_sets.append(X[test_index])

        return X_train_sets, X_test_sets, scaler, data_aggregated_with_cpu_ram
    else:
        return X, scaler, data_aggregated_with_cpu_ram


def split_features(data: np.ndarray, feature_dims: dict) -> List[np.ndarray]:
    """
    Splits data into duration, cpu, ram and time features based on feature dimensions.
    
    Args:
        data: Input data array
        feature_dims: Dictionary with feature dimensions
        
    Returns:
        List of feature arrays [duration_features, cpu_features, ram_features]
    """
    duration_dim = feature_dims['duration']
    cpu_dim = feature_dims['cpu']
    ram_dim = feature_dims['ram']
    time_dim = feature_dims['time']

    duration_features = data[:, :, :duration_dim]
    cpu_features = data[:, :, duration_dim:duration_dim + cpu_dim]
    ram_features = data[:, :, duration_dim + cpu_dim:duration_dim + cpu_dim + ram_dim]
    time_features = data[:, :, duration_dim + cpu_dim + ram_dim:duration_dim + cpu_dim + ram_dim + time_dim]

    return [duration_features, cpu_features, ram_features, time_features]


def get_train_test_split(traces_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, RobustScaler]:
    """
    Scales data and splits it into training and testing sets.
    
    Args:
        traces_df: DataFrame with trace data
        
    Returns:
        Tuple of (X_train, X_test, scaler)
    """
    traces_df = traces_df.drop(columns=['duration'])
    scaler = RobustScaler()
    X = scaler.fit_transform(traces_df)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    return X_train, X_test, scaler


def prepare_data_subtypes(data: pd.DataFrame) -> Tuple[np.ndarray, RobustScaler]:
    """
    Prepares the data for training by scaling features.
    
    Args:
        data: DataFrame with subtype data
        
    Returns:
        Tuple of (scaled data, scaler)
    """
    scaler = RobustScaler()
    X = scaler.fit_transform(data.drop(columns=['duration']))
    return X, scaler
