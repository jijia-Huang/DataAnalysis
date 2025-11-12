"""
資料載入工具模組

提供 CSV 檔案載入功能，支援多種編碼和分隔符號。
"""

import pandas as pd
import streamlit as st
from typing import Optional, Tuple


def load_csv_file(
    uploaded_file, 
    encodings: list = ['utf-8', 'big5', 'gb2312', 'latin-1'],
    delimiters: list = [',', ';', '\t']
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    載入 CSV 檔案，自動嘗試多種編碼和分隔符號。
    
    Args:
        uploaded_file: Streamlit 上傳的檔案物件
        encodings: 要嘗試的編碼列表，預設為 ['utf-8', 'big5', 'gb2312', 'latin-1']
        delimiters: 要嘗試的分隔符號列表，預設為 [',', ';', '\t']
    
    Returns:
        Tuple[Optional[DataFrame], Optional[str]]: 
            - 成功時返回 (DataFrame, None)
            - 失敗時返回 (None, 錯誤訊息)
    """
    if uploaded_file is None:
        return None, None
    
    # 讀取檔案內容
    file_content = uploaded_file.read()
    
    # 嘗試不同的編碼和分隔符號組合
    for encoding in encodings:
        try:
            # 解碼檔案內容
            decoded_content = file_content.decode(encoding)
            
            # 嘗試不同的分隔符號
            for delimiter in delimiters:
                try:
                    # 使用 StringIO 來讀取 CSV
                    from io import StringIO
                    df = pd.read_csv(StringIO(decoded_content), delimiter=delimiter)
                    
                    # 檢查是否成功讀取（至少有一列）
                    if df.shape[1] > 0:
                        return df, None
                except (pd.errors.ParserError, ValueError):
                    continue
        except UnicodeDecodeError:
            continue
        except Exception as e:
            # 如果解碼成功但讀取失敗，繼續嘗試下一個編碼
            continue
    
    return None, "無法讀取檔案。請確認檔案格式為 CSV，且編碼為 UTF-8 或 BIG5。"


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    驗證 DataFrame 的基本格式。
    
    Args:
        df: 要驗證的 DataFrame
    
    Returns:
        Tuple[bool, Optional[str]]: 
            - (True, None) 如果驗證通過
            - (False, 錯誤訊息) 如果驗證失敗
    """
    if df is None or df.empty:
        return False, "資料為空，請上傳包含資料的 CSV 檔案。"
    
    if df.shape[0] == 0:
        return False, "資料沒有行，請確認檔案內容。"
    
    if df.shape[1] == 0:
        return False, "資料沒有欄位，請確認檔案格式。"
    
    return True, None


def get_data_info(df: pd.DataFrame) -> dict:
    """
    取得資料的基本資訊。
    
    Args:
        df: DataFrame
    
    Returns:
        dict: 包含資料資訊的字典
    """
    if df is None or df.empty:
        return {}
    
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    return info

