"""
This module contains helper functions for preprocessing data for the
CNN-LSTM-Attention models.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from components.config import TARGET_THRESHOLD_LSTM
from signals._components.LSTM__function__create_classification_targets import create_classification_targets
from signals._components.LSTM__function__create_regression_targets import create_regression_targets

def create_target_variable(
    df: pd.DataFrame,
    output_mode: str,
    look_back: int
) -> pd.DataFrame:
    """
    Creates the target variable for the model.

    Args:
        df: The input DataFrame.
        output_mode: The type of target to create ('classification' or 'regression').
        look_back: The look-back period for target creation.

    Returns:
        The DataFrame with the 'target' column added.
    """
    df_with_target = df.copy()
    if output_mode == 'classification':
        df_with_target = create_classification_targets(
            df_with_target, future_shift=-look_back, threshold=TARGET_THRESHOLD_LSTM
        )
        # The function adds 'class_target', we rename it to 'target'
        if 'class_target' in df_with_target.columns:
            df_with_target.rename(columns={'class_target': 'target'}, inplace=True)

    elif output_mode == 'regression':
        df_with_target = create_regression_targets(
            df_with_target, future_shift=-look_back
        )
        # The function adds 'return_target', we rename it to 'target'
        if 'return_target' in df_with_target.columns:
            df_with_target.rename(columns={'return_target': 'target'}, inplace=True)
    else:
        raise ValueError(f"Invalid output_mode: {output_mode}. Must be 'classification' or 'regression'")

    df_with_target.dropna(subset=['target'], inplace=True)
    return df_with_target

def scale_data(
    df: pd.DataFrame,
    features: list[str],
    scaler_type: str = 'minmax'
) -> tuple[pd.DataFrame, Union[MinMaxScaler, StandardScaler]]:
    """
    Scales the specified features in the DataFrame.

    Args:
        df: The input DataFrame.
        features: The list of features to scale.
        scaler_type: The type of scaler to use ('minmax' or 'standard').

    Returns:
        A tuple containing the DataFrame with scaled features and the scaler object.
    """
    if scaler_type == 'minmax':
        scaler: Union[MinMaxScaler, StandardScaler] = MinMaxScaler(feature_range=(0, 1))
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Invalid scaler_type: {scaler_type}")

    df[features] = scaler.fit_transform(df[features])
    return df, scaler

def create_sequences(
    df: pd.DataFrame,
    features: List[str],
    look_back: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences of data for LSTM models.

    Args:
        df: The input DataFrame with features and target.
        features: The list of feature columns to use.
        look_back: The sequence length.

    Returns:
        A tuple containing the input sequences (X) and target values (y).
    """
    data = df[features + ['target']].values
    x_sequences, y_sequences = [], []
    for i in range(len(data) - look_back):
        x_sequences.append(data[i:(i + look_back), :-1])  # type: ignore
        y_sequences.append(data[i + look_back - 1, -1])   # type: ignore

    return np.array(x_sequences), np.array(y_sequences) 