"""
線性回歸梯度下降模型

使用 Scikit-learn 的 SGDRegressor 實作，支援記錄訓練過程。
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from typing import Any, Dict, List, Optional

from .base_model import BaseModel
from utils.data_preprocessor import preprocess_features, detect_categorical_features


class GradientDescentModel(BaseModel):
    """
    線性回歸梯度下降模型
    
    使用 Scikit-learn 的 SGDRegressor，支援不同的損失函式和訓練過程記錄。
    """
    
    # 損失函式對應表
    LOSS_FUNCTIONS = {
        'MSE': 'squared_error',  # 更新為新版本的損失函式名稱
        'MAE': 'epsilon_insensitive',
        'Huber': 'huber'
    }
    
    def __init__(self, loss: str = 'MSE', learning_rate: float = 0.01, 
                 max_iter: int = 1000, tol: float = 1e-6, use_scaling: bool = True):
        """
        初始化梯度下降模型
        
        Args:
            loss: 損失函式（'MSE', 'MAE', 'Huber'）
            learning_rate: 學習率
            max_iter: 最大迭代次數
            tol: 收斂容忍度
            use_scaling: 是否使用標準化（預設 True，強烈建議）
        """
        super().__init__("Gradient Descent")
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.use_scaling = use_scaling
        
        self.model: Optional[SGDRegressor] = None
        self.multi_output_model: Optional[MultiOutputRegressor] = None
        self.is_multi_output = False
        
        # 資料標準化器（梯度下降需要標準化資料）
        self.scaler: Optional[StandardScaler] = None
        
        # 類別型特徵編碼器
        self.encoder: Optional[OneHotEncoder] = None
        
        # 預處理元資料
        self.preprocessing_metadata: Dict[str, Any] = {}
        
        # 是否使用標準化
        self.use_scaling: bool = True
        
        # 訓練過程記錄
        self.training_history: List[Dict[str, Any]] = []
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, 
            record_history: bool = True, categorical_features: Optional[List[str]] = None, **kwargs) -> 'GradientDescentModel':
        """
        訓練梯度下降模型
        
        Args:
            X: 特徵資料（DataFrame）
            y: 目標變數（DataFrame，可以是單一或多個欄位）
            record_history: 是否記錄訓練過程（預設 True）
            categorical_features: 類別型特徵列表，如果為 None 則自動檢測
            **kwargs: 其他訓練參數
        
        Returns:
            self: 返回自身以支援鏈式呼叫
        """
        # 儲存特徵和目標變數名稱
        self.set_feature_names(list(X.columns))
        self.set_target_names(list(y.columns))
        
        # 判斷是否為多輸出迴歸
        self.is_multi_output = y.shape[1] > 1
        
        # 取得對應的損失函式名稱
        loss_function = self.LOSS_FUNCTIONS.get(self.loss, 'squared_error')
        
        # 重置訓練歷史
        self.training_history = []
        
        # 預處理特徵資料（獨熱編碼 + 標準化）
        X_processed, self.scaler, self.encoder, self.preprocessing_metadata = preprocess_features(
            X,
            categorical_features=categorical_features,  # 使用傳入的類別型特徵列表，如果為 None 則自動檢測
            use_scaling=self.use_scaling,
            fit=True
        )
        
        if self.is_multi_output:
            # 多輸出迴歸：使用 MultiOutputRegressor
            # 注意：SGDRegressor 的 partial_fit 在多輸出時需要分別處理每個目標
            base_model = SGDRegressor(
                loss=loss_function,
                learning_rate='constant',
                eta0=self.learning_rate,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=42
            )
            self.multi_output_model = MultiOutputRegressor(base_model)
            
            if record_history:
                # 使用 partial_fit 逐步訓練並記錄損失
                self._fit_with_history_multi(X_processed, y)
            else:
                self.multi_output_model.fit(X_processed.values, y.values)
            
            self.model = None
        else:
            # 單一輸出迴歸
            self.model = SGDRegressor(
                loss=loss_function,
                learning_rate='constant',
                eta0=self.learning_rate,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=42
            )
            
            if record_history:
                # 使用 partial_fit 逐步訓練並記錄損失
                self._fit_with_history_single(X_processed, y)
            else:
                self.model.fit(X_processed.values, y.values.ravel())
            
            self.multi_output_model = None
        
        self.is_trained = True
        return self
    
    def _fit_with_history_single(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        單一目標變數的逐步訓練並記錄歷史
        
        Args:
            X: 特徵資料
            y: 目標變數（單一欄位）
        """
        X_values = X.values
        y_values = y.values.ravel()
        
        # 初始化模型
        self.model.partial_fit(X_values, y_values)
        
        # 記錄初始損失
        y_pred = self.model.predict(X_values)
        mse = mean_squared_error(y_values, y_pred)
        mae = mean_absolute_error(y_values, y_pred)
        
        self.training_history.append({
            'iteration': 0,
            'loss': mse if self.loss == 'MSE' else mae,
            'mse': mse,
            'mae': mae
        })
        
        # 逐步訓練
        for iteration in range(1, self.max_iter):
            old_coef = self.model.coef_.copy()
            self.model.partial_fit(X_values, y_values)
            
            # 計算損失
            y_pred = self.model.predict(X_values)
            mse = mean_squared_error(y_values, y_pred)
            mae = mean_absolute_error(y_values, y_pred)
            
            current_loss = mse if self.loss == 'MSE' else mae
            
            self.training_history.append({
                'iteration': iteration,
                'loss': current_loss,
                'mse': mse,
                'mae': mae
            })
            
            # 檢查收斂
            if np.linalg.norm(self.model.coef_ - old_coef) < self.tol:
                break
    
    def _fit_with_history_multi(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        多目標變數的逐步訓練並記錄歷史
        
        Args:
            X: 特徵資料
            y: 目標變數（多個欄位）
        """
        X_values = X.values
        y_values = y.values
        
        # 初始化模型
        for i, estimator in enumerate(self.multi_output_model.estimators_):
            estimator.partial_fit(X_values, y_values[:, i])
        
        # 記錄初始損失
        y_pred = self.multi_output_model.predict(X_values)
        mse = np.mean([mean_squared_error(y_values[:, i], y_pred[:, i]) 
                      for i in range(y_values.shape[1])])
        mae = np.mean([mean_absolute_error(y_values[:, i], y_pred[:, i]) 
                      for i in range(y_values.shape[1])])
        
        self.training_history.append({
            'iteration': 0,
            'loss': mse if self.loss == 'MSE' else mae,
            'mse': mse,
            'mae': mae
        })
        
        # 逐步訓練
        for iteration in range(1, self.max_iter):
            old_coefs = [est.coef_.copy() for est in self.multi_output_model.estimators_]
            
            for i, estimator in enumerate(self.multi_output_model.estimators_):
                estimator.partial_fit(X_values, y_values[:, i])
            
            # 計算損失
            y_pred = self.multi_output_model.predict(X_values)
            mse = np.mean([mean_squared_error(y_values[:, i], y_pred[:, i]) 
                          for i in range(y_values.shape[1])])
            mae = np.mean([mean_absolute_error(y_values[:, i], y_pred[:, i]) 
                          for i in range(y_values.shape[1])])
            
            current_loss = mse if self.loss == 'MSE' else mae
            
            self.training_history.append({
                'iteration': iteration,
                'loss': current_loss,
                'mse': mse,
                'mae': mae
            })
            
            # 檢查收斂（檢查所有估計器的係數變化）
            converged = all(
                np.linalg.norm(self.multi_output_model.estimators_[i].coef_ - old_coefs[i]) < self.tol
                for i in range(len(self.multi_output_model.estimators_))
            )
            if converged:
                break
    
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
            use_scaling=self.use_scaling,
            fit=False,
            scaler=self.scaler,
            encoder=self.encoder
        )
        
        if self.is_multi_output:
            predictions = self.multi_output_model.predict(X_processed.values)
        else:
            predictions = self.model.predict(X_processed.values)
            predictions = predictions.reshape(-1, 1)
        
        # 轉換為 DataFrame
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
            'scaler': self.scaler,  # 保存標準化器
            'encoder': self.encoder,  # 保存編碼器
            'preprocessing_metadata': self.preprocessing_metadata,  # 保存預處理元資料
            'use_scaling': self.use_scaling,  # 保存標準化選項
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'model_name': self.model_name,
            'loss': self.loss,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, path)
    
    def load(self, path: str) -> 'GradientDescentModel':
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
        self.scaler = model_data.get('scaler')  # 載入標準化器
        self.encoder = model_data.get('encoder')  # 載入編碼器
        self.preprocessing_metadata = model_data.get('preprocessing_metadata', {})  # 載入預處理元資料
        self.use_scaling = model_data.get('use_scaling', True)  # 載入標準化選項
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.model_name = model_data['model_name']
        self.loss = model_data.get('loss', 'MSE')
        self.learning_rate = model_data.get('learning_rate', 0.01)
        self.max_iter = model_data.get('max_iter', 1000)
        self.tol = model_data.get('tol', 1e-6)
        self.training_history = model_data.get('training_history', [])
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
            'loss': self.loss,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'training_history_length': len(self.training_history)
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
                info['n_targets'] = len(self.target_names)
                info['n_features'] = len(self.feature_names)
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
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        取得訓練歷史記錄
        
        Returns:
            list: 訓練歷史記錄列表
        """
        return self.training_history

