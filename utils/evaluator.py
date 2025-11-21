"""
模型評估工具模組

提供單次評估和重複隨機分割評估功能。
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Any, Optional
import copy


def evaluate_model(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, float]:
    """
    計算模型的評估指標
    
    Args:
        y_true: 真實值（DataFrame）
        y_pred: 預測值（DataFrame）
    
    Returns:
        dict: 包含 R²、MSE、MAE 的字典
    """
    metrics = {}
    
    # 如果是多目標變數，計算每個目標的指標並取平均
    if y_true.shape[1] > 1:
        r2_scores = []
        mse_scores = []
        mae_scores = []
        
        for col in y_true.columns:
            if col in y_pred.columns:
                y_t = y_true[col].values
                y_p = y_pred[col].values
                
                r2_scores.append(r2_score(y_t, y_p))
                mse_scores.append(mean_squared_error(y_t, y_p))
                mae_scores.append(mean_absolute_error(y_t, y_p))
        
        metrics['R²'] = np.mean(r2_scores) if r2_scores else 0.0
        metrics['MSE'] = np.mean(mse_scores) if mse_scores else 0.0
        metrics['MAE'] = np.mean(mae_scores) if mae_scores else 0.0
    else:
        # 單一目標變數
        y_t = y_true.values.flatten()
        y_p = y_pred.values.flatten()
        
        metrics['R²'] = r2_score(y_t, y_p)
        metrics['MSE'] = mean_squared_error(y_t, y_p)
        metrics['MAE'] = mean_absolute_error(y_t, y_p)
    
    return metrics


def repeated_random_split_evaluate(
    model_class,
    model_params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_size: float = 0.2,
    n_repeats: int = 5,
    base_random_state: int = 42,
    progress_callback: Optional[callable] = None,
    categorical_features: Optional[List[str]] = None,
    augmentation_params: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
    """
    執行重複隨機分割評估
    
    Args:
        model_class: 模型類別（例如 LinearRegressionModel）
        model_params: 模型參數字典
        X: 特徵資料（DataFrame）
        y: 目標變數（DataFrame）
        test_size: 測試集比例
        n_repeats: 重複次數
        base_random_state: 基礎隨機種子
        progress_callback: 進度回調函數（可選），接收 (current, total) 參數
    
    Returns:
        Tuple: (平均值字典, 標準差字典, 每次評估結果列表)
    """
    if n_repeats < 3 or n_repeats > 200:
        raise ValueError(f"重複次數必須在 3 到 200 之間，當前值：{n_repeats}")
    
    if test_size <= 0 or test_size >= 1:
        raise ValueError(f"測試集比例必須在 0 到 1 之間，當前值：{test_size}")
    
    all_results = []
    
    for i in range(n_repeats):
        # 使用不同的隨機種子進行分割
        random_state = base_random_state + i
        
        # 隨機分割資料
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        
        # 資料擴增（如果啟用）
        if augmentation_params is not None:
            from utils.data_augmenter import augment_dataframe
            
            # 保存擴增前的資料
            X_train_before = X_train.copy()
            y_train_before = y_train.copy()
            
            # 擴增特徵資料
            X_train = augment_dataframe(
                X_train,
                noise_type=augmentation_params['noise_type'],
                noise_strength=augmentation_params['noise_strength'],
                multiplier=augmentation_params['multiplier'],
                random_state=random_state,  # 使用不同的隨機種子確保每次擴增不同
                categorical_features=categorical_features
            )
            
            # 擴增目標變數
            if y_train.select_dtypes(include=[np.number]).shape[1] > 0:
                # 如果目標變數是數值型，使用噪聲擴增
                y_train = augment_dataframe(
                    y_train_before,
                    noise_type=augmentation_params['noise_type'],
                    noise_strength=augmentation_params['noise_strength'],
                    multiplier=augmentation_params['multiplier'],
                    random_state=random_state
                )
            else:
                # 如果目標變數不是數值型，則重複對應次數
                y_train = pd.concat([y_train_before] * augmentation_params['multiplier'], ignore_index=True)
            
            # 確保 X_train 和 y_train 的行數一致
            if len(X_train) != len(y_train):
                min_len = min(len(X_train), len(y_train))
                X_train = X_train.iloc[:min_len]
                y_train = y_train.iloc[:min_len]
        
        # 創建並訓練模型
        model = model_class(**model_params)
        # 傳遞 categorical_features 參數（如果有的話）
        if categorical_features is not None:
            model.fit(X_train, y_train, categorical_features=categorical_features)
        else:
            model.fit(X_train, y_train)
        
        # 預測
        y_test_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        
        # 計算評估指標（測試集和訓練集）
        test_metrics = evaluate_model(y_test, y_test_pred)
        train_metrics = evaluate_model(y_train, y_train_pred)
        
        # 記錄詳細結果
        result = {
            '重複次數': i + 1,
            '隨機種子': random_state,
            '訓練集樣本數': len(X_train),
            '測試集樣本數': len(X_test),
            '訓練集 R²': train_metrics['R²'],
            '測試集 R²': test_metrics['R²'],
            '訓練集 MSE': train_metrics['MSE'],
            '測試集 MSE': test_metrics['MSE'],
            '訓練集 MAE': train_metrics['MAE'],
            '測試集 MAE': test_metrics['MAE'],
            'X_test': X_test.copy(),  # 保存測試集特徵
            'y_test': y_test.copy(),  # 保存測試集目標變數
            'y_test_pred': y_test_pred.copy()  # 保存測試集預測結果
        }
        all_results.append(result)
        
        # 調用進度回調
        if progress_callback:
            progress_callback(i + 1, n_repeats)
    
    # 計算平均值和標準差（僅針對測試集指標）
    metrics_names = ['R²', 'MSE', 'MAE']
    mean_metrics = {}
    std_metrics = {}
    
    for metric_name in metrics_names:
        test_key = f'測試集 {metric_name}'
        values = [result[test_key] for result in all_results]
        mean_metrics[metric_name] = np.mean(values)
        std_metrics[metric_name] = np.std(values)
    
    return mean_metrics, std_metrics, all_results


def format_metrics_with_std(mean_metrics: Dict[str, float], std_metrics: Dict[str, float]) -> pd.DataFrame:
    """
    格式化評估指標（平均值 ± 標準差）
    
    Args:
        mean_metrics: 平均值字典
        std_metrics: 標準差字典
    
    Returns:
        pd.DataFrame: 格式化的指標表格
    """
    formatted_data = []
    for metric_name in ['R²', 'MSE', 'MAE']:
        mean_val = mean_metrics.get(metric_name, 0.0)
        std_val = std_metrics.get(metric_name, 0.0)
        formatted_data.append({
            '指標': metric_name,
            '平均值': f"{mean_val:.6f}",
            '標準差': f"{std_val:.6f}",
            '平均值 ± 標準差': f"{mean_val:.6f} ± {std_val:.6f}"
        })
    
    return pd.DataFrame(formatted_data)

