"""
線性回歸模型

使用 Scikit-learn 的 LinearRegression 實作，支援正則化（L1/L2）。
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.multioutput import MultiOutputRegressor
import joblib
from typing import Any, Dict, Optional, List

from .base_model import BaseModel
from utils.data_preprocessor import preprocess_features


class LinearRegressionModel(BaseModel):
    """
    線性回歸模型
    
    使用 Scikit-learn 的 LinearRegression，支援單一和多個目標變數。
    支援正則化選項：無正則化、L1 (Lasso)、L2 (Ridge)。
    """
    
    def __init__(self, regularization: Optional[str] = None, alpha: float = 1.0):
        """
        初始化線性回歸模型
        
        Args:
            regularization: 正則化類型，可選值：None（無正則化）、'l1'（Lasso）、'l2'（Ridge）
            alpha: 正則化強度，預設為 1.0
        """
        super().__init__("Linear Regression")
        self.model: Optional[LinearRegression] = None
        self.multi_output_model: Optional[MultiOutputRegressor] = None
        self.is_multi_output = False
        
        # 正則化參數
        self.regularization = regularization  # None, 'l1', 'l2'
        self.alpha = alpha
        
        # 類別型特徵編碼器（線性回歸也需要處理類別型特徵）
        self.encoder = None
        self.preprocessing_metadata: Dict[str, Any] = {}
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, categorical_features: Optional[List[str]] = None, **kwargs) -> 'LinearRegressionModel':
        """
        訓練線性回歸模型
        
        Args:
            X: 特徵資料（DataFrame）
            y: 目標變數（DataFrame，可以是單一或多個欄位）
            categorical_features: 類別型特徵列表，如果為 None 則自動檢測
            **kwargs: 其他訓練參數（目前未使用）
        
        Returns:
            self: 返回自身以支援鏈式呼叫
        """
        # 儲存原始特徵名稱
        original_feature_names = list(X.columns)
        self.set_target_names(list(y.columns))
        
        # 預處理特徵資料（獨熱編碼類別型特徵）
        X_processed, _, self.encoder, self.preprocessing_metadata = preprocess_features(
            X,
            categorical_features=categorical_features,  # 使用傳入的類別型特徵列表，如果為 None 則自動檢測
            use_scaling=False,  # 線性回歸不需要標準化
            fit=True
        )
        
        # 儲存處理後的特徵名稱
        self.set_feature_names(list(X_processed.columns))
        
        # 判斷是否為多輸出迴歸
        self.is_multi_output = y.shape[1] > 1
        
        # 根據正則化類型選擇基礎模型
        if self.regularization == 'l1':
            base_model = Lasso(alpha=self.alpha)
        elif self.regularization == 'l2':
            base_model = Ridge(alpha=self.alpha)
        else:
            base_model = LinearRegression()
        
        if self.is_multi_output:
            # 多輸出迴歸：使用 MultiOutputRegressor
            self.multi_output_model = MultiOutputRegressor(base_model)
            self.multi_output_model.fit(X_processed.values, y.values)
            self.model = None
        else:
            # 單一輸出迴歸
            self.model = base_model
            self.model.fit(X_processed.values, y.values.ravel())
            self.multi_output_model = None
        
        self.is_trained = True
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        使用訓練好的模型進行預測
        
        Args:
            X: 特徵資料（DataFrame）
        
        Returns:
            DataFrame: 預測結果（與目標變數格式相同）
        """
        if not self.is_trained:
            raise ValueError("模型尚未訓練，請先呼叫 fit() 方法")
        
        # 使用相同的預處理流程轉換特徵資料
        X_processed, _, _, _ = preprocess_features(
            X,
            categorical_features=self.preprocessing_metadata.get('categorical_features'),
            use_scaling=False,  # 線性回歸不需要標準化
            fit=False,
            scaler=None,
            encoder=self.encoder
        )
        
        if self.is_multi_output:
            predictions = self.multi_output_model.predict(X_processed.values)
        else:
            predictions = self.model.predict(X_processed.values)
            predictions = predictions.reshape(-1, 1)
        
        # 轉換為 DataFrame，使用目標變數名稱作為欄位名稱
        result = pd.DataFrame(predictions, columns=self.target_names, index=X.index)
        return result
    
    def save(self, path: str) -> None:
        """
        保存模型到檔案
        
        Args:
            path: 保存路徑（應包含 .joblib 副檔名）
        """
        if not self.is_trained:
            raise ValueError("模型尚未訓練，無法保存")
        
        model_data = {
            'model': self.model,
            'multi_output_model': self.multi_output_model,
            'is_multi_output': self.is_multi_output,
            'encoder': self.encoder,  # 保存編碼器
            'preprocessing_metadata': self.preprocessing_metadata,  # 保存預處理元資料
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'model_name': self.model_name,
            'regularization': self.regularization,  # 保存正則化類型
            'alpha': self.alpha  # 保存正則化強度
        }
        
        joblib.dump(model_data, path)
    
    def load(self, path: str) -> 'LinearRegressionModel':
        """
        從檔案載入模型
        
        Args:
            path: 模型檔案路徑
        
        Returns:
            self: 返回自身以支援鏈式呼叫
        """
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.multi_output_model = model_data['multi_output_model']
        self.is_multi_output = model_data['is_multi_output']
        self.encoder = model_data.get('encoder')  # 載入編碼器
        self.preprocessing_metadata = model_data.get('preprocessing_metadata', {})  # 載入預處理元資料
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.model_name = model_data['model_name']
        # 載入正則化參數（向後相容：舊模型可能沒有這些欄位）
        self.regularization = model_data.get('regularization', None)  # 載入正則化類型
        self.alpha = model_data.get('alpha', 1.0)  # 載入正則化強度
        self.is_trained = True
        
        return self
    
    def get_info(self) -> Dict[str, Any]:
        """
        取得模型資訊
        
        Returns:
            dict: 包含模型類型、參數、目標變數、特徵等資訊的字典
        """
        info = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'is_multi_output': self.is_multi_output,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'regularization': self.regularization,  # 正則化類型
            'alpha': self.alpha if self.regularization else None  # 正則化強度（僅在有正則化時顯示）
        }
        
        # 加入特徵類型資訊（類別特徵和數值特徵）
        if self.preprocessing_metadata:
            info['categorical_features'] = self.preprocessing_metadata.get('categorical_features', [])
            info['numeric_features'] = self.preprocessing_metadata.get('numeric_features', [])
            info['original_columns'] = self.preprocessing_metadata.get('original_columns', [])
        else:
            info['categorical_features'] = []
            info['numeric_features'] = []
            info['original_columns'] = []
        
        if self.is_trained:
            if self.is_multi_output:
                # 多輸出模型的資訊
                info['n_targets'] = len(self.target_names)
                info['n_features'] = len(self.feature_names)
                # 取得每個目標的係數和截距
                coefficients = []
                intercepts = []
                for estimator in self.multi_output_model.estimators_:
                    coefficients.append(estimator.coef_.tolist())
                    intercepts.append(float(estimator.intercept_))
                info['coefficients'] = coefficients
                info['intercepts'] = intercepts
            else:
                # 單一輸出模型的資訊
                coef = self.model.coef_
                # 確保 coef 是一維數組
                if coef.ndim > 1:
                    coef = coef.flatten()
                info['coefficients'] = coef.tolist()
                info['intercept'] = float(self.model.intercept_)
                info['n_features'] = len(self.feature_names)
        
        return info

