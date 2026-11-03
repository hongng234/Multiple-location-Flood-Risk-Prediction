import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def split_data(df: pd.DataFrame, target: str, test_size: float, random_state: int):
    # Keep only the numeric columns for the feature matrix
    # (exclude date, location, and other string/datetime columns)
    feature_cols = df.select_dtypes(include=[np.number]).columns.drop(target, errors="ignore")
    X = df[feature_cols]
    y = df[target]

    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def build_preprocessor() -> Pipeline:
    return Pipeline([("scaler", StandardScaler())])