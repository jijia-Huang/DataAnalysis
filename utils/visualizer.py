"""
資料視覺化工具模組

提供資料分布和關係視覺化功能，使用 Plotly 建立互動式圖表。
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import List, Optional, Tuple


def create_histogram(df: pd.DataFrame, column: str, title: Optional[str] = None) -> go.Figure:
    """
    為單一數值欄位建立直方圖。
    
    Args:
        df: DataFrame
        column: 要繪製的欄位名稱
        title: 圖表標題，如果為 None 則使用欄位名稱
    
    Returns:
        go.Figure: Plotly 圖表物件
    """
    if title is None:
        title = f"{column} 的分布"
    
    fig = px.histogram(
        df, 
        x=column,
        title=title,
        labels={column: column},
        nbins=30
    )
    
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="頻率",
        showlegend=False
    )
    
    return fig


def create_histograms_for_numeric_columns(df: pd.DataFrame) -> List[go.Figure]:
    """
    為所有數值欄位建立直方圖。
    
    Args:
        df: DataFrame
    
    Returns:
        List[go.Figure]: 直方圖列表
    """
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not numeric_columns:
        return []
    
    figures = []
    for col in numeric_columns:
        fig = create_histogram(df, col)
        figures.append(fig)
    
    return figures


def create_scatter_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Optional[go.Figure]:
    """
    建立散點圖矩陣，顯示多個數值變數之間的關係。
    
    Args:
        df: DataFrame
        columns: 要包含的欄位列表，如果為 None 則使用所有數值欄位
    
    Returns:
        Optional[go.Figure]: Plotly 圖表物件，如果沒有足夠的數值欄位則返回 None
    """
    if columns is None:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    else:
        numeric_columns = [col for col in columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
    
    # 至少需要 2 個數值欄位才能建立散點圖矩陣
    if len(numeric_columns) < 2:
        return None
    
    # 限制最多顯示 6 個欄位，避免圖表過於複雜
    if len(numeric_columns) > 6:
        numeric_columns = numeric_columns[:6]
    
    try:
        fig = px.scatter_matrix(
            df,
            dimensions=numeric_columns,
            title="變數關係散點圖矩陣"
        )
        
        fig.update_layout(
            height=800,
            showlegend=False
        )
        
        return fig
    except Exception as e:
        # 如果建立失敗（例如資料量太大），返回 None
        return None


def create_statistics_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    建立統計摘要表格。
    
    Args:
        df: DataFrame
    
    Returns:
        pd.DataFrame: 統計摘要 DataFrame
    """
    return df.describe()


def plot_training_loss(training_history: List[dict], loss_type: str = 'MSE') -> go.Figure:
    """
    繪製訓練過程中的損失曲線。
    
    Args:
        training_history: 訓練歷史記錄列表，每個元素包含 iteration 和 loss
        loss_type: 損失類型（'MSE' 或 'MAE'），用於標題
    
    Returns:
        go.Figure: Plotly 圖表物件
    """
    if not training_history:
        # 如果沒有歷史記錄，返回空圖表
        fig = go.Figure()
        fig.add_annotation(
            text="沒有訓練歷史記錄",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    iterations = [h['iteration'] for h in training_history]
    losses = [h['loss'] for h in training_history]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iterations,
        y=losses,
        mode='lines+markers',
        name='損失',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title=f"訓練損失曲線 ({loss_type})",
        xaxis_title="迭代次數",
        yaxis_title=f"損失 ({loss_type})",
        hovermode='x unified',
        height=400
    )
    
    return fig


def display_model_parameters(model_info: dict) -> pd.DataFrame:
    """
    顯示模型參數（係數和截距）。
    
    Args:
        model_info: 模型資訊字典（來自 model.get_info()）
    
    Returns:
        pd.DataFrame: 模型參數表格
    """
    if not model_info.get('is_trained', False):
        return pd.DataFrame({'訊息': ['模型尚未訓練']})
    
    is_multi_output = model_info.get('is_multi_output', False)
    feature_names = model_info.get('feature_names', [])
    
    # 確保 feature_names 是列表且不為空
    if not feature_names or not isinstance(feature_names, list):
        feature_names = []
    
    if is_multi_output:
        # 多輸出模型
        coefficients = model_info.get('coefficients', [])
        intercepts = model_info.get('intercepts', [])
        target_names = model_info.get('target_names', [])
        
        # 確保所有係數列表長度一致
        if coefficients:
            # 檢查每個目標的係數長度
            for i, coef_list in enumerate(coefficients):
                if not isinstance(coef_list, list):
                    coefficients[i] = list(coef_list) if hasattr(coef_list, '__iter__') else [coef_list]
                # 確保係數長度與特徵名稱長度一致
                if len(coefficients[i]) != len(feature_names):
                    min_len = min(len(coefficients[i]), len(feature_names))
                    coefficients[i] = coefficients[i][:min_len]
                    if i == 0:  # 只在第一次調整 feature_names
                        feature_names = feature_names[:min_len]
        
        # 建立參數表格
        data = {'特徵': feature_names}
        for i, target in enumerate(target_names):
            if i < len(coefficients):
                coef_list = coefficients[i]
                # 確保長度一致
                if len(coef_list) != len(feature_names):
                    coef_list = coef_list[:len(feature_names)] if len(coef_list) > len(feature_names) else coef_list + [0] * (len(feature_names) - len(coef_list))
                data[f'{target}_係數'] = coef_list
            else:
                data[f'{target}_係數'] = [0] * len(feature_names)
        
        df = pd.DataFrame(data)
        
        # 添加截距行
        intercept_row = {'特徵': '截距'}
        for i, target in enumerate(target_names):
            intercept_row[f'{target}_係數'] = intercepts[i] if i < len(intercepts) else 0
        df = pd.concat([df, pd.DataFrame([intercept_row])], ignore_index=True)
        
    else:
        # 單一輸出模型
        coefficients = model_info.get('coefficients', [])
        intercept = model_info.get('intercept', 0)
        
        # 確保 coefficients 是列表格式
        if not isinstance(coefficients, list):
            coefficients = list(coefficients) if hasattr(coefficients, '__iter__') else [coefficients]
        
        # 確保 feature_names 和 coefficients 長度一致
        if len(feature_names) != len(coefficients):
            # 如果長度不一致，使用較短的長度
            min_len = min(len(feature_names), len(coefficients))
            if min_len == 0:
                # 如果長度為 0，返回錯誤信息
                return pd.DataFrame({
                    '錯誤': [f'特徵名稱數量 ({len(feature_names)}) 與係數數量 ({len(coefficients)}) 不一致']
                })
            feature_names = feature_names[:min_len]
            coefficients = coefficients[:min_len]
        
        df = pd.DataFrame({
            '特徵': feature_names + ['截距'],
            '係數': coefficients + [intercept]
        })
        
        # 格式化係數值，保留更多小數位以便查看
        df['係數'] = df['係數'].apply(lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else str(x))
    
    # 格式化多輸出模型的係數值
    if is_multi_output:
        for col in df.columns:
            if col != '特徵' and '_係數' in col:
                df[col] = df[col].apply(lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else str(x))
    
    return df


def diagnose_pvalue_issues(X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> dict:
    """
    診斷 p 值無法計算的原因
    
    Args:
        X: 特徵資料（numpy array）
        y: 目標變數（numpy array）
        feature_names: 特徵名稱列表（可選），用於顯示具體的特徵名稱
    
    Returns:
        dict: 診斷結果字典
    """
    n_samples, n_features = X.shape
    
    # 如果沒有提供特徵名稱，使用索引
    if feature_names is None:
        feature_names = [f"特徵_{i+1}" for i in range(n_features)]
    elif len(feature_names) != n_features:
        feature_names = [f"特徵_{i+1}" for i in range(n_features)]
    
    diagnostics = {
        'sample_feature_ratio': n_samples / n_features if n_features > 0 else 0,
        'n_samples': n_samples,
        'n_features': n_features,
        'degrees_of_freedom': n_samples - n_features - 1,
        'has_multicollinearity': False,
        'condition_number': None,
        'has_constant_features': False,
        'constant_feature_indices': [],
        'constant_feature_names': [],
        'highly_correlated_pairs': [],
        'issues': []
    }
    
    # 檢查樣本數量和特徵數量的比例
    if n_samples <= n_features:
        diagnostics['issues'].append(f"樣本數 ({n_samples}) 少於或等於特徵數 ({n_features})")
    
    if diagnostics['degrees_of_freedom'] <= 0:
        diagnostics['issues'].append(f"自由度不足 ({diagnostics['degrees_of_freedom']})")
    
    # 檢查是否有常數特徵
    if n_features > 0:
        constant_features = []
        constant_names = []
        for i in range(n_features):
            std_val = np.std(X[:, i])
            if std_val < 1e-10:
                constant_features.append(i)
                constant_names.append(feature_names[i])
        
        if constant_features:
            diagnostics['has_constant_features'] = True
            diagnostics['constant_feature_indices'] = constant_features
            diagnostics['constant_feature_names'] = constant_names
            diagnostics['issues'].append(f"發現 {len(constant_features)} 個常數特徵（變異數接近零）")
    
    # 檢查多重共線性
    try:
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        XtX = X_with_intercept.T @ X_with_intercept
        cond_number = np.linalg.cond(XtX)
        diagnostics['condition_number'] = cond_number
        
        if cond_number > 1e12:
            diagnostics['has_multicollinearity'] = True
            diagnostics['issues'].append(f"檢測到嚴重多重共線性（條件數: {cond_number:.2e}）")
        elif cond_number > 1e8:
            diagnostics['has_multicollinearity'] = True
            diagnostics['issues'].append(f"檢測到多重共線性（條件數: {cond_number:.2e}）")
        
        # 檢查特徵之間的相關性（排除常數特徵）
        if n_features > 1:
            # 計算相關性矩陣
            try:
                # 只檢查非常數特徵
                non_constant_indices = [i for i in range(n_features) if i not in diagnostics['constant_feature_indices']]
                if len(non_constant_indices) > 1:
                    X_non_constant = X[:, non_constant_indices]
                    corr_matrix = np.corrcoef(X_non_constant.T)
                    
                    # 找出高度相關的特徵對（相關係數 > 0.95）
                    highly_correlated = []
                    for i in range(len(non_constant_indices)):
                        for j in range(i+1, len(non_constant_indices)):
                            corr_val = corr_matrix[i, j]
                            if abs(corr_val) > 0.95:
                                idx_i = non_constant_indices[i]
                                idx_j = non_constant_indices[j]
                                highly_correlated.append({
                                    'feature1': feature_names[idx_i],
                                    'feature2': feature_names[idx_j],
                                    'correlation': corr_val
                                })
                    
                    diagnostics['highly_correlated_pairs'] = highly_correlated
                    
                    if highly_correlated:
                        diagnostics['issues'].append(f"發現 {len(highly_correlated)} 對高度相關的特徵（|r| > 0.95）")
            except Exception:
                # 如果計算相關性失敗，忽略
                pass
                
    except Exception:
        diagnostics['issues'].append("無法計算條件數（矩陣可能不可逆）")
    
    return diagnostics


def calculate_p_values(X: np.ndarray, y: np.ndarray, coefficients: np.ndarray, intercept: float) -> Tuple[np.ndarray, float]:
    """
    計算線性回歸係數的 p 值
    
    Args:
        X: 特徵資料（numpy array，不包含常數項）
        y: 目標變數（numpy array）
        coefficients: 回歸係數（numpy array，對應 X 的特徵）
        intercept: 截距
    
    Returns:
        Tuple: (係數的 p 值陣列, 截距的 p 值)
    """
    n_samples, n_features = X.shape
    
    # 檢查樣本數量和特徵數量的比例
    if n_samples <= n_features:
        # 樣本數少於或等於特徵數，無法計算 p 值
        return np.full(n_features, np.nan), np.nan
    
    # 添加常數項到 X（用於計算截距的標準誤差）
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    
    # 計算預測值
    y_pred = X @ coefficients + intercept
    
    # 計算殘差
    residuals = y.flatten() - y_pred.flatten()
    
    # 計算殘差平方和（RSS）
    rss = np.sum(residuals ** 2)
    
    # 計算自由度（n - p - 1，其中 p 是特徵數量）
    degrees_of_freedom = n_samples - n_features - 1
    
    if degrees_of_freedom <= 0:
        # 如果自由度不足，返回 NaN
        return np.full(n_features, np.nan), np.nan
    
    # 計算殘差標準誤差（Residual Standard Error）
    mse = rss / degrees_of_freedom  # 均方誤差
    rmse = np.sqrt(mse)  # 根均方誤差
    
    # 檢查是否有殘差為零或接近零（完美擬合）
    if rss < 1e-10:
        # 如果殘差平方和接近零，標準誤差可能不準確
        return np.full(n_features, np.nan), np.nan
    
    # 計算 (X'X) 矩陣
    XtX = X_with_intercept.T @ X_with_intercept
    
    # 檢查矩陣的條件數（用於診斷多重共線性）
    try:
        cond_number = np.linalg.cond(XtX)
        # 條件數過大表示存在多重共線性
        if cond_number > 1e12:
            # 嘗試使用偽逆（pseudo-inverse）作為備選方案
            XtX_inv = np.linalg.pinv(XtX)
        else:
            # 正常情況下使用標準逆矩陣
            XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # 如果標準逆矩陣失敗，嘗試使用偽逆
        try:
            XtX_inv = np.linalg.pinv(XtX)
        except:
            # 如果偽逆也失敗，返回 NaN
            return np.full(n_features, np.nan), np.nan
    
    # 計算所有參數（包括截距）的標準誤差
    try:
        diag_XtX_inv = np.diag(XtX_inv)
        # 檢查是否有負值或無效值
        if np.any(diag_XtX_inv < 0) or np.any(np.isnan(diag_XtX_inv)) or np.any(np.isinf(diag_XtX_inv)):
            return np.full(n_features, np.nan), np.nan
        
        se_all = rmse * np.sqrt(diag_XtX_inv)
        
        # 檢查標準誤差是否有效
        if np.any(np.isnan(se_all)) or np.any(np.isinf(se_all)) or np.any(se_all <= 0):
            return np.full(n_features, np.nan), np.nan
        
        # 分離截距和係數的標準誤差
        se_intercept = se_all[0]
        se_coefficients = se_all[1:]
        
    except Exception:
        # 如果計算標準誤差失敗，返回 NaN
        return np.full(n_features, np.nan), np.nan
    
    # 計算 t 統計量和 p 值
    p_values = np.full(n_features, np.nan)
    
    for i in range(n_features):
        if not np.isnan(se_coefficients[i]) and se_coefficients[i] > 1e-10:  # 避免除零
            try:
                t_stat = coefficients[i] / se_coefficients[i]
                if not np.isnan(t_stat) and not np.isinf(t_stat) and np.abs(t_stat) < 1e10:
                    p_values[i] = 2 * (1 - stats.t.cdf(np.abs(t_stat), degrees_of_freedom))
                    # 確保 p 值在合理範圍內
                    if p_values[i] < 0 or p_values[i] > 1:
                        p_values[i] = np.nan
            except:
                p_values[i] = np.nan
        else:
            p_values[i] = np.nan
    
    # 計算截距的 t 統計量和 p 值
    if not np.isnan(se_intercept) and se_intercept > 1e-10:
        try:
            t_intercept = intercept / se_intercept
            if not np.isnan(t_intercept) and not np.isinf(t_intercept) and np.abs(t_intercept) < 1e10:
                p_intercept = 2 * (1 - stats.t.cdf(np.abs(t_intercept), degrees_of_freedom))
                if p_intercept < 0 or p_intercept > 1:
                    p_intercept = np.nan
            else:
                p_intercept = np.nan
        except:
            p_intercept = np.nan
    else:
        p_intercept = np.nan
    
    return p_values, p_intercept


def display_model_parameters_with_pvalues(
    model_info: dict, 
    X: np.ndarray, 
    y: np.ndarray
) -> pd.DataFrame:
    """
    顯示模型參數（係數和截距）以及 p 值
    
    Args:
        model_info: 模型資訊字典（來自 model.get_info()）
        X: 特徵資料（numpy array，需要包含常數項）
        y: 目標變數（numpy array）
    
    Returns:
        pd.DataFrame: 包含係數、p 值和顯著性的表格
    """
    if not model_info.get('is_trained', False):
        return pd.DataFrame({'訊息': ['模型尚未訓練']})
    
    is_multi_output = model_info.get('is_multi_output', False)
    feature_names = model_info.get('feature_names', [])
    
    if not feature_names or not isinstance(feature_names, list):
        feature_names = []
    
    if is_multi_output:
        # 多輸出模型暫不支持 p 值計算
        return display_model_parameters(model_info)
    else:
        # 單一輸出模型
        coefficients = np.array(model_info.get('coefficients', []))
        intercept = model_info.get('intercept', 0)
        
        # 計算 p 值（X 不包含常數項，截距單獨處理）
        try:
            # 確保輸入格式正確
            X_array = X if isinstance(X, np.ndarray) else np.array(X)
            y_array = y.flatten() if y.ndim > 1 else (y if isinstance(y, np.ndarray) else np.array(y))
            coef_array = coefficients if isinstance(coefficients, np.ndarray) else np.array(coefficients)
            
            # 檢查維度匹配
            if X_array.shape[1] != len(coef_array):
                raise ValueError(f"特徵數量 ({X_array.shape[1]}) 與係數數量 ({len(coef_array)}) 不匹配")
            
            p_coefficients, p_intercept_val = calculate_p_values(
                X_array, 
                y_array,
                coef_array,
                intercept
            )
        except Exception as e:
            # 如果計算失敗，返回沒有 p 值的表格
            p_intercept_val = np.nan
            p_coefficients = np.full(len(coefficients), np.nan)
            # 可以選擇性地記錄錯誤（用於調試）
            import warnings
            warnings.warn(f"p 值計算失敗: {str(e)}")
        
        # 檢查是否有任何有效的 p 值
        has_valid_pvalues = not np.all(np.isnan(p_coefficients)) or not np.isnan(p_intercept_val)
        
        # 確保長度一致
        min_len = min(len(feature_names), len(coefficients), len(p_coefficients))
        if min_len > 0:
            feature_names = feature_names[:min_len]
            coefficients = coefficients[:min_len]
            p_coefficients = p_coefficients[:min_len]
        
        # 判斷顯著性
        def get_significance(p_val):
            if np.isnan(p_val):
                return "N/A"
            elif p_val < 0.001:
                return "***"
            elif p_val < 0.01:
                return "**"
            elif p_val < 0.05:
                return "*"
            elif p_val < 0.1:
                return "."
            else:
                return ""
        
        df = pd.DataFrame({
            '特徵': feature_names + ['截距'],
            '係數': list(coefficients) + [intercept],
            'p 值': list(p_coefficients) + [p_intercept_val],
            '顯著性': [get_significance(p) for p in list(p_coefficients) + [p_intercept_val]]
        })
        
        # 格式化數值
        df['係數'] = df['係數'].apply(lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else str(x))
        df['p 值'] = df['p 值'].apply(lambda x: f"{x:.6f}" if not np.isnan(x) else "N/A")
        
        return df


def display_evaluation_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """
    計算並顯示評估指標。
    
    Args:
        y_true: 真實值（DataFrame）
        y_pred: 預測值（DataFrame）
    
    Returns:
        pd.DataFrame: 評估指標表格
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    metrics = []
    
    for col in y_true.columns:
        if col in y_pred.columns:
            y_t = y_true[col].values
            y_p = y_pred[col].values
            
            r2 = r2_score(y_t, y_p)
            mse = mean_squared_error(y_t, y_p)
            mae = mean_absolute_error(y_t, y_p)
            
            metrics.append({
                '目標變數': col,
                'R²': round(r2, 4),
                'MSE': round(mse, 4),
                'MAE': round(mae, 4)
            })
    
    return pd.DataFrame(metrics)

