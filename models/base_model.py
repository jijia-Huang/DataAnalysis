"""
模型抽象基類

定義所有機器學習模型的統一介面，確保模型可以替換和擴展。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    模型抽象基類
    
    所有模型都應該繼承此類別並實作所有抽象方法。
    這確保了統一的介面，讓模型可以輕鬆替換。
    """
    
    def __init__(self, model_name: str):
        """
        初始化模型
        
        Args:
            model_name: 模型名稱
        """
        self.model_name = model_name
        self.is_trained = False
        self.feature_names: Optional[list] = None
        self.target_names: Optional[list] = None
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'BaseModel':
        """
        訓練模型
        
        Args:
            X: 特徵資料（DataFrame）
            y: 目標變數（DataFrame，可以是單一或多個欄位）
            **kwargs: 其他訓練參數
        
        Returns:
            self: 返回自身以支援鏈式呼叫
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        使用訓練好的模型進行預測
        
        Args:
            X: 特徵資料（DataFrame）
        
        Returns:
            DataFrame: 預測結果（與目標變數格式相同）
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        保存模型到檔案
        
        Args:
            path: 保存路徑
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> 'BaseModel':
        """
        從檔案載入模型
        
        Args:
            path: 模型檔案路徑
        
        Returns:
            self: 返回自身以支援鏈式呼叫
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        取得模型資訊
        
        Returns:
            dict: 包含模型類型、參數、目標變數、特徵等資訊的字典
        """
        pass
    
    def set_feature_names(self, feature_names: list) -> None:
        """
        設定特徵名稱
        
        Args:
            feature_names: 特徵名稱列表
        """
        self.feature_names = feature_names
    
    def set_target_names(self, target_names: list) -> None:
        """
        設定目標變數名稱
        
        Args:
            target_names: 目標變數名稱列表
        """
        self.target_names = target_names

