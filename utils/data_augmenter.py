"""
資料擴增工具模組

提供數值型資料的噪聲擴增功能，支援高斯噪聲和均勻噪聲。
"""

import pandas as pd
import numpy as np
from typing import Tuple, Literal, Optional, List


def add_gaussian_noise(data: pd.Series, noise_strength: float, random_state: int = None) -> pd.Series:
    """
    為數值欄位添加高斯噪聲
    
    Args:
        data: 要添加噪聲的資料序列
        noise_strength: 噪聲強度（相對於標準差的比例）
        random_state: 隨機種子，用於可重複性
    
    Returns:
        pd.Series: 添加噪聲後的資料
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    std = data.std()
    if std == 0 or pd.isna(std):
        # 如果標準差為 0 或 NaN，不添加噪聲
        return data.copy()
    
    noise_std = std * noise_strength
    noise = np.random.normal(0, noise_std, size=len(data))
    return data + noise


def add_uniform_noise(data: pd.Series, noise_strength: float, random_state: int = None) -> pd.Series:
    """
    為數值欄位添加均勻噪聲
    
    Args:
        data: 要添加噪聲的資料序列
        noise_strength: 噪聲強度（相對於標準差的比例）
        random_state: 隨機種子，用於可重複性
    
    Returns:
        pd.Series: 添加噪聲後的資料
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    std = data.std()
    if std == 0 or pd.isna(std):
        # 如果標準差為 0 或 NaN，不添加噪聲
        return data.copy()
    
    noise_range = std * noise_strength
    noise = np.random.uniform(-noise_range, noise_range, size=len(data))
    return data + noise


def augment_dataframe(
    df: pd.DataFrame,
    noise_type: Literal['gaussian', 'uniform'] = 'gaussian',
    noise_strength: float = 0.1,
    multiplier: int = 2,
    random_state: int = None,
    categorical_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    對 DataFrame 進行資料擴增
    
    僅對數值型欄位添加噪聲，類別型欄位保持不變。
    
    Args:
        df: 要擴增的 DataFrame
        noise_type: 噪聲類型，'gaussian' 或 'uniform'
        noise_strength: 噪聲強度（相對於標準差的比例），範圍 0.01-0.5
        multiplier: 擴增倍數，範圍 1-5（2 表示將資料擴增為原來的 2 倍）
        random_state: 隨機種子，用於可重複性
        categorical_features: 類別型特徵列表，這些特徵不會添加噪聲
    
    Returns:
        pd.DataFrame: 擴增後的 DataFrame
    
    Raises:
        ValueError: 當參數無效時
    """
    # 參數驗證
    if noise_strength < 0.01 or noise_strength > 0.5:
        raise ValueError(f"噪聲強度必須在 0.01 到 0.5 之間，當前值：{noise_strength}")
    
    if multiplier < 1 or multiplier > 5:
        raise ValueError(f"擴增倍數必須在 1 到 5 之間，當前值：{multiplier}")
    
    if noise_type not in ['gaussian', 'uniform']:
        raise ValueError(f"噪聲類型必須是 'gaussian' 或 'uniform'，當前值：{noise_type}")
    
    if df.empty:
        return df.copy()
    
    # 識別數值型欄位
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 排除類別型特徵（即使它們是數值型）
    if categorical_features:
        categorical_features = [f for f in categorical_features if f in df.columns]
        numeric_columns = [col for col in numeric_columns if col not in categorical_features]
    
    if not numeric_columns:
        # 如果沒有數值型欄位，直接返回原始資料
        return df.copy()
    
    # 複製原始資料
    augmented_dfs = [df.copy()]
    
    # 生成擴增樣本
    for i in range(multiplier - 1):
        augmented_df = df.copy()
        
        # 為每個數值欄位添加噪聲
        for col in numeric_columns:
            if noise_type == 'gaussian':
                augmented_df[col] = add_gaussian_noise(
                    df[col], 
                    noise_strength, 
                    random_state=random_state + i if random_state is not None else None
                )
            else:  # uniform
                augmented_df[col] = add_uniform_noise(
                    df[col], 
                    noise_strength, 
                    random_state=random_state + i if random_state is not None else None
                )
        
        augmented_dfs.append(augmented_df)
    
    # 合併所有擴增樣本
    result = pd.concat(augmented_dfs, ignore_index=True)
    
    return result


def get_augmentation_stats(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame
) -> pd.DataFrame:
    """
    計算擴增前後的統計資訊
    
    Args:
        df_before: 擴增前的 DataFrame
        df_after: 擴增後的 DataFrame
    
    Returns:
        pd.DataFrame: 包含統計資訊的 DataFrame
    """
    numeric_columns = df_before.select_dtypes(include=[np.number]).columns.tolist()
    
    stats = []
    for col in numeric_columns:
        stats.append({
            '欄位': col,
            '擴增前樣本數': len(df_before),
            '擴增後樣本數': len(df_after),
            '擴增前平均值': df_before[col].mean(),
            '擴增後平均值': df_after[col].mean(),
            '擴增前標準差': df_before[col].std(),
            '擴增後標準差': df_after[col].std()
        })
    
    return pd.DataFrame(stats)

