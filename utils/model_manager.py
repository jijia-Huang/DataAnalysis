"""
模型管理工具

提供模型保存、載入、列表查詢和刪除功能。
"""

import os
import json
import joblib
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

from models.base_model import BaseModel
from models.linear_regression import LinearRegressionModel
from models.gradient_descent import GradientDescentModel


MODELS_DIR = Path("models")


def ensure_models_directory() -> None:
    """確保 models 目錄存在"""
    MODELS_DIR.mkdir(exist_ok=True)


def save_model(model: BaseModel, model_name: str) -> str:
    """
    保存模型到檔案，同時保存欄位資訊到同檔名的 .json 檔案
    
    Args:
        model: 要保存的模型實例
        model_name: 模型名稱（不含副檔名）
    
    Returns:
        str: 保存的檔案路徑
    """
    ensure_models_directory()
    
    # 加入時間戳記確保唯一性
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.joblib"
    filepath = MODELS_DIR / filename
    
    # 保存模型
    model.save(str(filepath))
    
    # 取得模型資訊並保存到 JSON 檔案
    model_info = model.get_info()
    
    # 提取欄位資訊
    field_info = {
        'model_name': model_info.get('model_name', 'Unknown'),
        'target_names': model_info.get('target_names', []),
        'feature_names': model_info.get('feature_names', []),
        'original_columns': model_info.get('original_columns', []),
        'categorical_features': model_info.get('categorical_features', []),
        'numeric_features': model_info.get('numeric_features', []),
        'is_multi_output': model_info.get('is_multi_output', False)
    }
    
    # 保存到同檔名的 .json 檔案
    json_filepath = filepath.with_suffix('.json')
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(field_info, f, ensure_ascii=False, indent=2)
    
    return str(filepath)


def load_model(filepath: str) -> BaseModel:
    """
    從檔案載入模型
    
    Args:
        filepath: 模型檔案路徑
    
    Returns:
        BaseModel: 載入的模型實例
    """
    # 先載入模型資料以判斷模型類型
    model_data = joblib.load(filepath)
    
    model_name = model_data.get('model_name', 'Unknown')
    
    # 根據模型名稱判斷類型
    if 'Gradient Descent' in model_name:
        model = GradientDescentModel()
    elif 'Linear Regression' in model_name:
        model = LinearRegressionModel()
    else:
        # 預設嘗試線性回歸
        model = LinearRegressionModel()
    
    model.load(filepath)
    return model


def list_models() -> List[Dict[str, Any]]:
    """
    列出所有已保存的模型
    
    Returns:
        list: 模型資訊列表，每個元素包含檔案名稱、路徑、修改時間等
    """
    ensure_models_directory()
    
    models = []
    for filepath in MODELS_DIR.glob("*.joblib"):
        try:
            # 載入模型以取得資訊
            model = load_model(str(filepath))
            info = model.get_info()
            
            # 取得檔案修改時間
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            
            models.append({
                'filename': filepath.name,
                'filepath': str(filepath),
                'model_name': info.get('model_name', 'Unknown'),
                'model_type': info.get('model_name', 'Unknown'),
                'is_trained': info.get('is_trained', False),
                'target_names': info.get('target_names', []),
                'feature_names': info.get('feature_names', []),
                'modified_time': mtime,
                'size': filepath.stat().st_size
            })
        except Exception as e:
            # 如果載入失敗，仍然列出檔案但標記為錯誤
            models.append({
                'filename': filepath.name,
                'filepath': str(filepath),
                'model_name': 'Error loading model',
                'error': str(e)
            })
    
    # 按修改時間排序（最新的在前）
    models.sort(key=lambda x: x.get('modified_time', datetime.min), reverse=True)
    
    return models


def delete_model(filepath: str) -> bool:
    """
    刪除模型檔案及其對應的 JSON 欄位資訊檔案
    
    Args:
        filepath: 模型檔案路徑
    
    Returns:
        bool: 是否成功刪除
    """
    try:
        path = Path(filepath)
        if path.exists() and path.suffix == '.joblib':
            # 刪除模型檔案
            path.unlink()
            
            # 刪除對應的 JSON 檔案（如果存在）
            json_path = path.with_suffix('.json')
            if json_path.exists():
                json_path.unlink()
            
            return True
        return False
    except Exception:
        return False


def get_model_info(filepath: str) -> Optional[Dict[str, Any]]:
    """
    取得模型資訊（不載入完整模型）
    
    Args:
        filepath: 模型檔案路徑
    
    Returns:
        dict: 模型資訊，如果載入失敗則返回 None
    """
    try:
        model = load_model(filepath)
        return model.get_info()
    except Exception:
        return None


def load_field_info(filepath: str) -> Optional[Dict[str, Any]]:
    """
    從 JSON 檔案載入欄位資訊（不需要載入完整模型）
    
    Args:
        filepath: 模型檔案路徑（.joblib 或 .json）
    
    Returns:
        dict: 欄位資訊，如果檔案不存在或載入失敗則返回 None
    """
    try:
        # 如果是 .joblib 檔案，轉換為 .json 檔案路徑
        path = Path(filepath)
        if path.suffix == '.joblib':
            json_path = path.with_suffix('.json')
        elif path.suffix == '.json':
            json_path = path
        else:
            # 嘗試添加 .json 副檔名
            json_path = path.with_suffix('.json')
        
        if not json_path.exists():
            return None
        
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

