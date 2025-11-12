"""
資料預處理工具

提供資料標準化和類別型特徵編碼功能。
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, Optional, List, Dict, Any


def convert_thousand_separator_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    檢測並轉換帶有千分位分隔符的欄位為數值
    
    Args:
        df: DataFrame
    
    Returns:
        Tuple: (轉換後的 DataFrame, 已轉換的欄位名稱列表)
    """
    df_converted = df.copy()
    converted_columns = []
    
    for col in df.columns:
        # 如果欄位已經是數值型，跳過
        if df[col].dtype in [np.number, 'int64', 'int32', 'float64', 'float32']:
            continue
        
        # 檢查資料是否包含千分位分隔符（逗號）
        try:
            sample_values = df[col].dropna().astype(str).head(100)
            has_comma = sample_values.str.contains(',', na=False).any()
            
            if has_comma:
                # 轉換帶有千分位分隔符的字串為數值
                def convert_thousand_separator(value):
                    if pd.isna(value):
                        return np.nan
                    value_str = str(value).strip()
                    # 移除千分位分隔符（逗號）和其他可能的符號
                    value_str = value_str.replace(',', '').replace(' ', '')
                    try:
                        return float(value_str)
                    except (ValueError, TypeError):
                        return np.nan
                
                df_converted[col] = df[col].apply(convert_thousand_separator)
                converted_columns.append(col)
        except Exception:
            # 如果轉換失敗，跳過這個欄位
            pass
    
    return df_converted, converted_columns


def convert_percentage_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    檢測並轉換百分比格式的欄位為數值
    
    Args:
        df: DataFrame
    
    Returns:
        Tuple: (轉換後的 DataFrame, 已轉換的欄位名稱列表)
    """
    df_converted = df.copy()
    converted_columns = []
    
    for col in df.columns:
        # 檢查欄位名稱是否包含「占比」、「百分比」等關鍵字，或檢查資料是否為百分比格式
        is_percentage_col = (
            '占比' in col or 
            '百分比' in col or 
            'percent' in col.lower() or 
            'percentage' in col.lower()
        )
        
        # 如果欄位名稱暗示是百分比，或資料中包含 % 符號
        if is_percentage_col or df[col].dtype == 'object':
            # 嘗試轉換百分比格式
            try:
                # 檢查是否包含百分比符號
                sample_values = df[col].dropna().astype(str).head(100)
                has_percentage = sample_values.str.contains('%', na=False).any()
                
                if has_percentage or is_percentage_col:
                    # 轉換百分比字串為數值（例如 "50%" -> 0.5）
                    def convert_percentage(value):
                        if pd.isna(value):
                            return np.nan
                        value_str = str(value).strip()
                        # 移除 % 符號並轉換為數值
                        value_str = value_str.replace('%', '').strip()
                        try:
                            num_value = float(value_str)
                            # 如果數值大於 1，假設是 0-100 格式，轉換為 0-1
                            if num_value > 1:
                                return num_value / 100.0
                            # 否則假設已經是 0-1 格式
                            return num_value
                        except (ValueError, TypeError):
                            return np.nan
                    
                    df_converted[col] = df[col].apply(convert_percentage)
                    converted_columns.append(col)
            except Exception:
                # 如果轉換失敗，跳過這個欄位
                pass
    
    return df_converted, converted_columns


def detect_categorical_features(df: pd.DataFrame, max_unique: int = 20) -> List[str]:
    """
    檢測類別型特徵
    
    Args:
        df: DataFrame
        max_unique: 最大唯一值數量，超過此數值視為數值型特徵
    
    Returns:
        list: 類別型特徵名稱列表
    """
    categorical_features = []
    
    for col in df.columns:
        # 檢查資料類型
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_features.append(col)
        # 檢查整數型但唯一值較少的欄位（可能是類別型）
        elif df[col].dtype in ['int64', 'int32', 'int16', 'int8']:
            unique_count = df[col].nunique()
            if unique_count <= max_unique and unique_count < len(df) * 0.5:
                # 如果唯一值數量少且不是大部分都是唯一值，可能是類別型
                categorical_features.append(col)
        # 檢查字串型（即使不在數值欄位中）
        elif df[col].dtype == 'string':
            categorical_features.append(col)
    
    return categorical_features


def preprocess_features(
    X: pd.DataFrame,
    categorical_features: Optional[List[str]] = None,
    use_scaling: bool = True,
    fit: bool = True,
    scaler: Optional[StandardScaler] = None,
    encoder: Optional[OneHotEncoder] = None
) -> Tuple[pd.DataFrame, StandardScaler, OneHotEncoder, Dict[str, Any]]:
    """
    預處理特徵資料
    
    Args:
        X: 特徵資料（DataFrame）
        categorical_features: 類別型特徵列表，如果為 None 則自動檢測
        use_scaling: 是否進行標準化
        fit: 是否擬合編碼器和標準化器（True 用於訓練，False 用於預測）
        scaler: 標準化器（預測時使用）
        encoder: 獨熱編碼器（預測時使用）
    
    Returns:
        Tuple: (處理後的 DataFrame, scaler, encoder, 元資料)
    """
    X_processed = X.copy()
    metadata = {
        'original_columns': list(X.columns),
        'categorical_features': [],
        'numeric_features': [],
        'encoded_feature_names': [],
        'percentage_columns': [],
        'thousand_separator_columns': []
    }
    
    # 先轉換千分位分隔符欄位為數值
    X_processed, thousand_separator_columns = convert_thousand_separator_columns(X_processed)
    metadata['thousand_separator_columns'] = thousand_separator_columns
    
    # 再轉換百分比欄位為數值
    X_processed, percentage_columns = convert_percentage_columns(X_processed)
    metadata['percentage_columns'] = percentage_columns
    
    # 檢測類別型特徵（在轉換百分比之後）
    if categorical_features is None:
        categorical_features = detect_categorical_features(X_processed)
    
    metadata['categorical_features'] = categorical_features
    metadata['numeric_features'] = [col for col in X.columns if col not in categorical_features]
    
    # 處理類別型特徵：獨熱編碼
    if categorical_features:
        if fit:
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(X[categorical_features])
        else:
            if encoder is None:
                raise ValueError("預測時必須提供 encoder")
            encoded_data = encoder.transform(X[categorical_features])
        
        # 建立編碼後的欄位名稱
        encoded_feature_names = []
        for i, col in enumerate(categorical_features):
            categories = encoder.categories_[i]
            # 跳過第一個類別（因為 drop='first'）
            for cat in categories[1:]:
                encoded_feature_names.append(f"{col}_{cat}")
        
        metadata['encoded_feature_names'] = encoded_feature_names
        
        # 移除原始類別型特徵
        X_processed = X_processed.drop(columns=categorical_features)
        
        # 添加編碼後的特徵
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoded_feature_names,
            index=X.index
        )
        X_processed = pd.concat([X_processed, encoded_df], axis=1)
    else:
        # 沒有類別型特徵時，encoder 為 None
        if not fit and encoder is not None:
            # 預測時如果提供了 encoder 但沒有類別型特徵，這是異常情況
            pass  # 忽略 encoder
    
    # 標準化數值型特徵
    if use_scaling and len(metadata['numeric_features']) > 0:
        numeric_cols = metadata['numeric_features']
        
        if fit:
            scaler = StandardScaler()
            X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
        else:
            if scaler is None:
                raise ValueError("預測時必須提供 scaler")
            X_processed[numeric_cols] = scaler.transform(X_processed[numeric_cols])
    
    return X_processed, scaler, encoder, metadata

