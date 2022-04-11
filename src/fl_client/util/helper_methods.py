"""
    Provides helper methods
"""
from typing import Dict
import pandas as pd


def pop_labels(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    """
    Retrieves labels and puts them into a dict
    """
    return {bearing: df.pop("RUL") for bearing, df in df_dict.items()}
