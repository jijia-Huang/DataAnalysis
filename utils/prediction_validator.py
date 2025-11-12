"""
預測資料驗證工具

提供預測輸入資料的驗證功能，確保輸入資料與模型特徵一致。
"""

import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from models.base_model import BaseModel


def validate_feature_names(
    input_df: pd.DataFrame,
    model: BaseModel
) -> Tuple[bool, Optional[str], Optional[List[str]]]:
    """
    驗證輸入資料的特徵名稱是否與模型一致
    
    Args:
        input_df: 輸入資料 DataFrame
        model: 訓練好的模型
    
    Returns:
        Tuple[bool, Optional[str], Optional[List[str]]]: 
            (是否通過驗證, 錯誤訊息, 目標變數列表)
    """
    if model.feature_names is None:
        return False, "模型沒有特徵名稱資訊", None
    
    # 取得目標變數名稱
    target_names = model.target_names or []
    
    # 取得原始特徵名稱（從 preprocessing_metadata）
    preprocessing_metadata = getattr(model, 'preprocessing_metadata', {})
    original_columns = preprocessing_metadata.get('original_columns', [])
    
    if not original_columns:
        # 如果沒有原始欄位資訊，使用 feature_names（可能是編碼後的）
        required_features = model.feature_names
    else:
        # 使用原始欄位名稱（編碼前的）
        required_features = original_columns
    
    # 檢查輸入資料是否包含目標變數（這不應該發生，但我們要處理）
    input_targets = [col for col in input_df.columns if col in target_names]
    
    # 檢查是否包含所有必需的特徵
    missing_features = [f for f in required_features if f not in input_df.columns]
    
    if missing_features:
        error_msg = f"缺少以下特徵欄位：{', '.join(missing_features)}"
        if input_targets:
            error_msg += f"\n\n注意：輸入資料包含目標變數欄位（{', '.join(input_targets)}），這些欄位不應該在預測輸入中。"
        return False, error_msg, input_targets
    
    # 如果通過驗證但有目標變數，返回警告資訊
    if input_targets:
        return True, None, input_targets
    
    return True, None, []


def validate_data_types(
    input_df: pd.DataFrame,
    model: BaseModel
) -> Tuple[bool, Optional[str]]:
    """
    驗證輸入資料的資料類型是否與模型一致
    
    Args:
        input_df: 輸入資料 DataFrame
        model: 訓練好的模型
    
    Returns:
        Tuple[bool, Optional[str]]: (是否通過驗證, 錯誤訊息)
    """
    preprocessing_metadata = getattr(model, 'preprocessing_metadata', {})
    categorical_features = preprocessing_metadata.get('categorical_features', [])
    numeric_features = preprocessing_metadata.get('numeric_features', [])
    
    errors = []
    
    # 檢查數值型特徵
    for feature in numeric_features:
        if feature in input_df.columns:
            if not pd.api.types.is_numeric_dtype(input_df[feature]):
                # 嘗試轉換為數值型
                try:
                    pd.to_numeric(input_df[feature], errors='raise')
                except (ValueError, TypeError):
                    errors.append(f"特徵 '{feature}' 應該是數值型，但包含非數值資料")
    
    # 檢查類別型特徵
    for feature in categorical_features:
        if feature in input_df.columns:
            # 類別型特徵可以是任何類型，但需要檢查值是否在訓練時見過
            pass  # 類別值驗證在另一個函數中處理
    
    if errors:
        return False, "; ".join(errors)
    
    return True, None


def validate_categorical_values(
    input_df: pd.DataFrame,
    model: BaseModel
) -> Tuple[bool, Optional[str], Optional[Dict[str, List[str]]]]:
    """
    驗證類別型特徵的值是否在訓練時見過
    
    Args:
        input_df: 輸入資料 DataFrame
        model: 訓練好的模型
    
    Returns:
        Tuple[bool, Optional[str], Optional[Dict]]: 
            (是否通過驗證, 錯誤訊息, 有效類別值字典)
    """
    encoder = getattr(model, 'encoder', None)
    preprocessing_metadata = getattr(model, 'preprocessing_metadata', {})
    categorical_features = preprocessing_metadata.get('categorical_features', [])
    
    if not categorical_features or encoder is None:
        return True, None, None
    
    # 取得每個類別型特徵的有效值
    valid_categories = {}
    if hasattr(encoder, 'categories_'):
        for i, feature in enumerate(categorical_features):
            if i < len(encoder.categories_):
                valid_categories[feature] = [str(cat) for cat in encoder.categories_[i]]
    
    errors = []
    unknown_values = {}
    
    for feature in categorical_features:
        if feature in input_df.columns:
            # 取得輸入資料中的唯一值
            input_values = input_df[feature].dropna().unique()
            valid_values = valid_categories.get(feature, [])
            
            # 檢查是否有未見過的值
            unknown = [str(val) for val in input_values if str(val) not in valid_values]
            
            if unknown:
                unknown_values[feature] = unknown
                errors.append(
                    f"特徵 '{feature}' 包含未見過的值：{', '.join(unknown[:5])}"
                    + (f" 等（共 {len(unknown)} 個）" if len(unknown) > 5 else "")
                )
    
    if errors:
        error_msg = "; ".join(errors)
        if valid_categories:
            error_msg += "\n\n有效值：\n"
            for feature, values in valid_categories.items():
                error_msg += f"  - {feature}: {', '.join(values[:10])}"
                if len(values) > 10:
                    error_msg += f" 等（共 {len(values)} 個）"
                error_msg += "\n"
        
        return False, error_msg, valid_categories
    
    return True, None, valid_categories


def validate_missing_values(
    input_df: pd.DataFrame,
    model: BaseModel
) -> Tuple[bool, Optional[str]]:
    """
    驗證輸入資料是否有缺失值
    
    Args:
        input_df: 輸入資料 DataFrame
        model: 訓練好的模型
    
    Returns:
        Tuple[bool, Optional[str]]: (是否通過驗證, 錯誤訊息)
    """
    preprocessing_metadata = getattr(model, 'preprocessing_metadata', {})
    original_columns = preprocessing_metadata.get('original_columns', [])
    
    if not original_columns:
        required_features = model.feature_names or []
    else:
        required_features = original_columns
    
    # 檢查必需特徵是否有缺失值
    missing_features = []
    for feature in required_features:
        if feature in input_df.columns:
            if input_df[feature].isnull().any():
                missing_features.append(feature)
    
    if missing_features:
        return False, f"以下特徵包含缺失值：{', '.join(missing_features)}。請填補缺失值後再進行預測。"
    
    return True, None


def validate_prediction_input(
    input_df: pd.DataFrame,
    model: BaseModel
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    完整驗證預測輸入資料
    
    Args:
        input_df: 輸入資料 DataFrame
        model: 訓練好的模型
    
    Returns:
        Tuple[bool, Optional[str], Optional[Dict]]: 
            (是否通過驗證, 錯誤訊息, 驗證詳細資訊)
    """
    if not model.is_trained:
        return False, "模型尚未訓練，無法進行預測", None
    
    validation_info = {}
    
    # 1. 驗證特徵名稱
    is_valid, error, target_vars_in_input = validate_feature_names(input_df, model)
    if not is_valid:
        return False, error, None
    validation_info['feature_names'] = "通過"
    if target_vars_in_input:
        validation_info['target_vars_in_input'] = target_vars_in_input
    
    # 2. 驗證缺失值
    is_valid, error = validate_missing_values(input_df, model)
    if not is_valid:
        return False, error, None
    validation_info['missing_values'] = "通過"
    
    # 3. 驗證資料類型
    is_valid, error = validate_data_types(input_df, model)
    if not is_valid:
        return False, error, None
    validation_info['data_types'] = "通過"
    
    # 4. 驗證類別值
    is_valid, error, valid_categories = validate_categorical_values(input_df, model)
    if not is_valid:
        return False, error, {'valid_categories': valid_categories}
    validation_info['categorical_values'] = "通過"
    
    return True, None, validation_info

