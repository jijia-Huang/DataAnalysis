"""
訓練頁面

提供模型訓練功能，支援線性回歸和梯度下降訓練。
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.data_loader import load_csv_file, validate_dataframe
from utils.model_manager import save_model, list_models
from utils.data_preprocessor import (
    detect_categorical_features, 
    convert_percentage_columns,
    convert_thousand_separator_columns
)
from utils.visualizer import (
    plot_training_loss,
    display_model_parameters,
    display_model_parameters_with_pvalues,
    display_evaluation_metrics
)
from models.linear_regression import LinearRegressionModel
from models.gradient_descent import GradientDescentModel


st.title("🎯 模型訓練")
st.markdown("---")

# 說明文字
st.info(
    "👋 歡迎使用模型訓練功能！\n\n"
    "請按照以下步驟訓練模型：\n"
    "1. 選擇或上傳訓練資料\n"
    "2. 選擇目標變數（要預測的欄位）\n"
    "3. 選擇訓練演算法和參數\n"
    "4. 開始訓練並查看結果\n"
    "5. 保存訓練好的模型"
)

# ========== 步驟 1: 資料選擇 ==========
st.markdown("### 步驟 1: 選擇訓練資料")

# 優先檢查 session state 中是否有資料分析頁面的資料
df = None
data_source = "session"

if 'data_analysis_df' in st.session_state and st.session_state['data_analysis_df'] is not None:
    df = st.session_state['data_analysis_df']
    st.success(f"✅ 已從資料分析頁面載入資料（{df.shape[0]} 筆，{df.shape[1]} 欄位）")
    st.dataframe(df.head(5), use_container_width=True)
    
    use_existing = st.checkbox("使用此資料", value=True)
    if not use_existing:
        df = None
        data_source = "upload"

if df is None:
    data_source = "upload"
    uploaded_file = st.file_uploader(
        "或上傳新的 CSV 檔案",
        type=['csv'],
        help="請選擇一個 CSV 格式的資料檔案"
    )
    
    if uploaded_file is not None:
        with st.spinner("正在載入資料..."):
            df, error_message = load_csv_file(uploaded_file)
            
            if error_message:
                st.error(f"❌ {error_message}")
                st.stop()
            
            is_valid, validation_error = validate_dataframe(df)
            
            if not is_valid:
                st.error(f"❌ {validation_error}")
                st.stop()
            
            st.success("✅ 資料載入成功！")
            st.dataframe(df.head(5), use_container_width=True)

if df is None:
    st.warning("👆 請先選擇或上傳資料")
    st.stop()

st.markdown("---")

# ========== 步驟 2: 排除欄位選擇 ==========
st.markdown("### 步驟 2: 選擇要排除的欄位（選填）")

# 取得所有欄位（包括非數值欄位）
all_columns = df.columns.tolist()

# 初始化排除欄位列表
excluded_columns = []

if all_columns:
    excluded_columns = st.multiselect(
        "選擇要排除的欄位（不會用於訓練）",
        options=all_columns,
        help="可以選擇一個或多個不需要用於訓練的欄位，例如 ID、索引、日期等。這些欄位將不會作為特徵變數或目標變數。"
    )
    
    if excluded_columns:
        # 顯示排除的欄位資訊
        excluded_numeric = [col for col in excluded_columns if col in df.select_dtypes(include=[np.number]).columns.tolist()]
        excluded_non_numeric = [col for col in excluded_columns if col not in excluded_numeric]
        
        info_parts = []
        if excluded_numeric:
            info_parts.append(f"{len(excluded_numeric)} 個數值欄位")
        if excluded_non_numeric:
            info_parts.append(f"{len(excluded_non_numeric)} 個非數值欄位")
        
        st.info(f"✅ 已排除 {len(excluded_columns)} 個欄位（{', '.join(info_parts)}）：{', '.join(excluded_columns)}")
else:
    st.info("ℹ️ 沒有可排除的欄位")

st.markdown("---")

# ========== 步驟 3: 資料預處理設定 ==========
st.markdown("### 步驟 3: 資料預處理設定")

# 取得可用欄位（排除已排除的欄位）
available_columns = [col for col in df.columns if col not in excluded_columns]

if not available_columns:
    st.error("❌ 沒有可用的欄位（所有欄位都已被排除）。")
    st.stop()

# 對所有可用欄位進行預處理預覽
df_preview = df[available_columns].copy()

# 先轉換千分位分隔符欄位為數值
df_preprocessed, thousand_separator_columns_detected = convert_thousand_separator_columns(df_preview)

# 再轉換百分比欄位為數值
df_preprocessed, percentage_columns_detected = convert_percentage_columns(df_preprocessed)

# 檢測類別型特徵（在轉換千分位和百分比之後）
categorical_features_detected = detect_categorical_features(df_preprocessed)

# 顯示千分位分隔符欄位資訊
if thousand_separator_columns_detected:
    st.success(
        f"🔢 檢測到 {len(thousand_separator_columns_detected)} 個帶有千分位分隔符的欄位：\n\n"
        f"**{', '.join(thousand_separator_columns_detected)}**\n\n"
        "✅ 這些欄位已轉換為數值格式（例如 \"1,000\" → 1000）。"
    )

# 顯示百分比欄位資訊
if percentage_columns_detected:
    st.success(
        f"📊 檢測到 {len(percentage_columns_detected)} 個百分比欄位：\n\n"
        f"**{', '.join(percentage_columns_detected)}**\n\n"
        "✅ 這些欄位將自動轉換為數值格式（例如 \"50%\" → 0.5）。"
    )

if categorical_features_detected:
    st.success(
        f"🔍 檢測到 {len(categorical_features_detected)} 個類別型特徵：\n\n"
        f"**{', '.join(categorical_features_detected)}**\n\n"
        "✅ 這些特徵將自動使用**獨熱編碼（One-Hot Encoding）**處理。"
    )
    
    # 顯示類別型特徵的詳細資訊
    with st.expander("📋 查看類別型特徵詳細資訊"):
        for cat_feat in categorical_features_detected:
            unique_values = df_preview[cat_feat].unique()
            unique_count = len(unique_values)
            st.write(f"**{cat_feat}**: {unique_count} 個唯一值")
            st.write(f"  值: {', '.join([str(v) for v in unique_values[:10]])}")
            if unique_count > 10:
                st.write(f"  ... 還有 {unique_count - 10} 個值")
            st.write("---")
else:
    st.info("ℹ️ 未檢測到類別型特徵，所有特徵將作為數值型處理。")

# 將預處理後的資料保存到 session state，供後續使用
st.session_state['preprocessed_df'] = df_preprocessed
st.session_state['thousand_separator_columns'] = thousand_separator_columns_detected
st.session_state['percentage_columns'] = percentage_columns_detected
st.session_state['categorical_features'] = categorical_features_detected

st.markdown("---")

# ========== 步驟 4: 目標變數選擇 ==========
st.markdown("### 步驟 4: 選擇目標變數")

# 從預處理後的資料中取得所有數值欄位（包括轉換後的百分比欄位）
# 使用 session state 中的預處理資料
if 'preprocessed_df' not in st.session_state:
    st.error("❌ 請先完成資料預處理設定。")
    st.stop()
df_preprocessed = st.session_state['preprocessed_df']
numeric_columns = df_preprocessed.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_columns:
    st.error("❌ 預處理後的資料中沒有數值欄位，無法進行訓練。")
    st.stop()

# 目標變數選擇（多選）
target_variables = st.multiselect(
    "選擇要預測的欄位（目標變數）",
    options=numeric_columns,
    help="可以選擇一個或多個欄位作為目標變數。選擇多個目標變數時，模型將同時預測所有目標。注意：百分比欄位已轉換為數值，可以選擇作為目標變數。"
)

if not target_variables:
    st.warning("👆 請至少選擇一個目標變數")
    st.stop()

# 顯示選擇的目標變數資訊
if len(target_variables) > 1:
    st.success(f"✅ 已選擇 {len(target_variables)} 個目標變數：{', '.join(target_variables)}")
    st.info("💡 多目標變數預測：模型將同時學習預測所有選定的目標變數。損失函數將分別應用到每個目標變數。")
else:
    st.info(f"✅ 已選擇目標變數：{target_variables[0]}")

st.markdown("---")

# ========== 步驟 5: 訓練演算法選擇 ==========
st.markdown("### 步驟 5: 選擇訓練演算法")

algorithm = st.radio(
    "選擇訓練演算法",
    options=["線性回歸", "線性回歸梯度下降"],
    help="線性回歸：快速穩定，使用最小二乘法\n線性回歸梯度下降：可顯示訓練過程，支援不同損失函式"
)

# 如果選擇了多個目標變數，顯示損失函數說明
if len(target_variables) > 1:
    st.info(
        "💡 **多目標變數損失函數說明**：\n\n"
        "當預測多個目標變數時，損失函數會分別應用到每個目標變數：\n"
        "- **MSE（均方誤差）**：對每個目標變數分別計算 MSE，然後取平均\n"
        "- **MAE（平均絕對誤差）**：對每個目標變數分別計算 MAE，然後取平均\n"
        "- **Huber**：對每個目標變數分別計算 Huber 損失，然後取平均\n\n"
        "模型會同時優化所有目標變數的預測效果。"
    )

if algorithm == "線性回歸梯度下降":
    use_scaling = st.checkbox(
        "使用資料標準化",
        value=True,
        help="標準化可以改善梯度下降的收斂速度和效果。強烈建議開啟（預設開啟）。"
    )
else:
    use_scaling = False

st.markdown("---")

# ========== 步驟 6: 參數設定 ==========
st.markdown("### 步驟 6: 設定訓練參數")

# 初始化正則化變數（確保在所有情況下都有定義）
regularization_type = "無正則化"
alpha = 1.0

# 線性回歸的正則化選項
if algorithm == "線性回歸":
    regularization_type = st.radio(
        "正則化類型",
        options=["無正則化", "L1 (Lasso)", "L2 (Ridge)"],
        help="正則化可以幫助防止過擬合，提高模型的泛化能力"
    )
    
    # 正則化強度設定（僅在選擇 L1 或 L2 時顯示）
    if regularization_type != "無正則化":
        alpha = st.slider(
            "正則化強度 (alpha)",
            min_value=0.001,
            max_value=100.0,
            value=1.0,
            step=0.1,
            format="%.3f",
            help="alpha 值越大，正則化效果越強。建議從 1.0 開始嘗試。"
        )
    
    # 可展開/收起的正則化說明
    with st.expander("📖 正則化說明（點擊展開）"):
        st.markdown("""
        **什麼是正則化？**
        
        正則化是一種防止模型過擬合的技術。當模型過於複雜時，可能會過度學習訓練資料的細節，導致在新資料上表現不佳。正則化通過添加懲罰項來控制模型的複雜度。
        
        **L1 正則化 (Lasso)**
        - **作用**：會將某些特徵的係數縮減為 0，實現特徵選擇
        - **特點**：適合當您認為只有部分特徵重要時使用
        - **效果**：可以自動排除不重要的特徵，簡化模型
        
        **L2 正則化 (Ridge)**
        - **作用**：會縮小所有特徵的係數，但不會完全消除
        - **特點**：適合當您認為所有特徵都有一定重要性時使用
        - **效果**：讓模型參數更平滑，減少極端值
        
        **如何選擇？**
        - **選擇 L1**：當您想要自動選擇重要特徵，或特徵數量很多時
        - **選擇 L2**：當您想要保留所有特徵，但希望模型更穩定時
        - **無正則化**：當資料量足夠大，或不需要控制過擬合時
        """)
    
    st.markdown("---")

if algorithm == "線性回歸梯度下降":
    col1, col2 = st.columns(2)
    
    with col1:
        loss_function = st.selectbox(
            "損失函式",
            options=["MSE", "MAE", "Huber"],
            help="MSE：均方誤差（常用）\nMAE：平均絕對誤差\nHuber：對異常值較不敏感"
        )
        
        learning_rate = st.number_input(
            "學習率",
            min_value=0.0001,
            max_value=1.0,
            value=0.01,
            step=0.001,
            help="學習率控制每次更新的步長"
        )
    
    with col2:
        max_iter = st.number_input(
            "最大迭代次數",
            min_value=10,
            max_value=10000,
            value=1000,
            step=10,
            help="訓練的最大迭代次數"
        )
        
        tol = st.number_input(
            "收斂容忍度",
            min_value=1e-10,
            max_value=1e-3,
            value=1e-6,
            format="%e",
            help="當損失變化小於此值時停止訓練"
        )
else:
    loss_function = None
    learning_rate = None
    max_iter = None
    tol = None

st.markdown("---")

# ========== 步驟 7: 資料分割設定 ==========
st.markdown("### 步驟 7: 資料分割設定")

split_data = st.checkbox(
    "分割訓練集和測試集",
    value=False,
    help="選擇是否將資料分割為訓練集和測試集。分割後，模型將在訓練集上訓練，並在測試集上評估。"
)

test_size = 0.2  # 預設測試集比例
if split_data:
    test_size = st.slider(
        "測試集比例",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="測試集佔總資料的比例（例如 0.2 表示 20% 的資料作為測試集）"
    )
    st.info(f"📊 資料將分割為：訓練集 {int((1-test_size)*100)}%，測試集 {int(test_size*100)}%")
    
    # 評估方式選擇（僅在分割資料時顯示）
    evaluation_method = st.radio(
        "評估方式",
        options=["single", "repeated"],
        format_func=lambda x: "單次評估" if x == "single" else "重複隨機分割評估",
        help="單次評估：快速評估，適合快速測試\n重複隨機分割評估：多次評估取平均，結果更穩定可靠"
    )
    
    n_repeats = 5  # 預設重複次數
    if evaluation_method == "repeated":
        n_repeats = st.number_input(
            "重複次數",
            min_value=3,
            max_value=20,
            value=5,
            step=1,
            help="重複評估的次數。次數越多，結果越穩定，但需要更長的時間"
        )
        st.info(
            "💡 **重複隨機分割評估說明**：\n\n"
            "- **作用**：透過多次隨機分割和評估來獲得更穩定、可靠的評估結果\n"
            "- **優點**：減少單次隨機分割的偏差，提供評估指標的變異性資訊\n"
            "- **結果**：顯示平均值 ± 標準差，讓您了解評估結果的穩定性\n"
            "- **注意**：重複評估會增加訓練時間（約 N 倍），請根據資料大小選擇合適的重複次數"
        )
else:
    # 如果沒有分割資料，使用單次評估
    evaluation_method = "single"
    n_repeats = 1

st.markdown("---")

# ========== 步驟 8: 資料擴增設定 ==========
st.markdown("### 步驟 8: 資料擴增設定（可選）")

enable_augmentation = st.checkbox(
    "啟用資料擴增",
    value=False,
    help="資料擴增可以透過對數值欄位添加噪聲來增加訓練樣本數量，有助於提升模型的泛化能力"
)

augmentation_params = {}
if enable_augmentation:
    st.info(
        "💡 **資料擴增說明**：\n\n"
        "- **作用**：透過對數值欄位添加適度的噪聲來增加訓練樣本數量\n"
        "- **適用欄位**：僅對數值型欄位進行擴增，類別型欄位保持不變\n"
        "- **擴增範圍**：僅對訓練集進行擴增，測試集保持原樣以確保評估準確性\n"
        "- **建議**：當訓練資料量較少時，可以使用資料擴增來提升模型效果"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        noise_type = st.radio(
            "噪聲類型",
            options=["gaussian", "uniform"],
            format_func=lambda x: "高斯噪聲（Gaussian）" if x == "gaussian" else "均勻噪聲（Uniform）",
            help="高斯噪聲：符合常態分布，適合大多數情況\n均勻噪聲：在固定範圍內均勻分布"
        )
        
        noise_strength = st.slider(
            "噪聲強度",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="噪聲強度相對於欄位標準差的比例。值越大，添加的噪聲越多"
        )
    
    with col2:
        multiplier = st.number_input(
            "擴增倍數",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            help="將訓練資料擴增為原來的幾倍。例如：2 倍表示資料量翻倍"
        )
    
    augmentation_params = {
        'noise_type': noise_type,
        'noise_strength': noise_strength,
        'multiplier': multiplier
    }

st.markdown("---")

# ========== 步驟 9: 訓練執行 ==========
st.markdown("### 步驟 9: 開始訓練")

if st.button("🚀 開始訓練", type="primary", use_container_width=True):
    with st.spinner("正在訓練模型..."):
        try:
            # 準備資料
            # 使用預處理後的資料（從 session state 取得）
            if 'preprocessed_df' not in st.session_state:
                st.error("❌ 請先完成資料預處理設定。")
                st.stop()
            df_processed = st.session_state['preprocessed_df']
            
            # 特徵變數 = 所有可用欄位 - 目標變數
            feature_columns = [
                col for col in df_processed.columns 
                if col not in target_variables
            ]
            
            if not feature_columns:
                st.error("❌ 沒有可用的特徵變數。請確保至少有一個欄位作為特徵變數（排除目標變數）。")
                st.stop()
            
            X = df_processed[feature_columns].copy()
            y = df_processed[target_variables].copy()
            
            # 顯示使用的特徵變數資訊
            st.info(f"📊 使用 {len(feature_columns)} 個特徵變數：{', '.join(feature_columns)}")
            
            # 處理缺失值（簡單策略：移除）
            X = X.dropna()
            y = y.loc[X.index]
            
            if len(X) == 0:
                st.error("❌ 資料中缺失值過多，無法進行訓練。")
                st.stop()
            
            # 資料分割（如果需要）
            X_train = X
            y_train = y
            X_test = None
            y_test = None
            
            if split_data:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    random_state=42,
                    shuffle=True
                )
                st.success(f"✅ 資料已分割：訓練集 {len(X_train)} 筆，測試集 {len(X_test)} 筆")
            
            # 資料擴增（如果啟用）
            if enable_augmentation:
                from utils.data_augmenter import augment_dataframe, get_augmentation_stats
                
                st.markdown("#### 📈 執行資料擴增")
                with st.spinner("正在擴增訓練資料..."):
                    try:
                        # 保存擴增前的資料用於統計
                        X_train_before = X_train.copy()
                        y_train_before = y_train.copy()
                        
                        # 取得類別型特徵列表（如果存在）
                        categorical_features_for_aug = st.session_state.get('categorical_features', None)
                        # 確保類別型特徵只包含在 X_train 中的特徵
                        if categorical_features_for_aug:
                            categorical_features_for_aug = [f for f in categorical_features_for_aug if f in X_train.columns]
                        
                        # 擴增特徵資料（排除類別型特徵）
                        X_train = augment_dataframe(
                            X_train,
                            noise_type=augmentation_params['noise_type'],
                            noise_strength=augmentation_params['noise_strength'],
                            multiplier=augmentation_params['multiplier'],
                            random_state=42,
                            categorical_features=categorical_features_for_aug
                        )
                        
                        # 擴增目標變數（確保與 X_train 的行數一致）
                        # 由於 X_train 已經擴增，y_train 需要對應擴增
                        if y_train.select_dtypes(include=[np.number]).shape[1] > 0:
                            # 如果目標變數是數值型，使用噪聲擴增
                            y_train = augment_dataframe(
                                y_train_before,
                                noise_type=augmentation_params['noise_type'],
                                noise_strength=augmentation_params['noise_strength'],
                                multiplier=augmentation_params['multiplier'],
                                random_state=42
                            )
                        else:
                            # 如果目標變數不是數值型，則重複對應次數
                            y_train = pd.concat([y_train_before] * augmentation_params['multiplier'], ignore_index=True)
                        
                        # 確保 X_train 和 y_train 的行數一致
                        if len(X_train) != len(y_train):
                            # 如果行數不一致，調整 y_train 以匹配 X_train
                            min_len = min(len(X_train), len(y_train))
                            X_train = X_train.iloc[:min_len]
                            y_train = y_train.iloc[:min_len]
                        
                        # 顯示擴增統計資訊
                        st.success(f"✅ 資料擴增完成：訓練集從 {len(X_train_before)} 筆擴增到 {len(X_train)} 筆")
                        
                        # 顯示統計資訊
                        if len(X_train_before.select_dtypes(include=[np.number]).columns) > 0:
                            stats_df = get_augmentation_stats(X_train_before, X_train)
                            st.markdown("**擴增前後統計資訊：**")
                            st.dataframe(stats_df, use_container_width=True)
                    
                    except Exception as e:
                        st.warning(f"⚠️ 資料擴增失敗：{str(e)}\n\n將使用原始資料進行訓練。")
                        # 如果擴增失敗，使用原始資料
                        X_train = X_train_before if 'X_train_before' in locals() else X_train
                        y_train = y_train_before if 'y_train_before' in locals() else y_train
            
            # 根據選擇的演算法建立模型
            # 取得檢測到的類別型特徵（如果存在）
            categorical_features = st.session_state.get('categorical_features', None)
            # 確保類別型特徵只包含在 X_train 中的特徵
            if categorical_features:
                categorical_features = [f for f in categorical_features if f in X_train.columns]
            
            if algorithm == "線性回歸":
                # 根據選擇的正則化類型設定參數
                regularization = None
                if regularization_type == "L1 (Lasso)":
                    regularization = 'l1'
                elif regularization_type == "L2 (Ridge)":
                    regularization = 'l2'
                
                model = LinearRegressionModel(
                    regularization=regularization,
                    alpha=alpha if regularization else 1.0
                )
                model.fit(X_train, y_train, categorical_features=categorical_features)
                
                # 顯示預處理資訊
                preprocessing_info = []
                if model.preprocessing_metadata.get('thousand_separator_columns'):
                    ts_features = model.preprocessing_metadata['thousand_separator_columns']
                    preprocessing_info.append(f"✅ {len(ts_features)} 個千分位分隔符欄位已轉換為數值：{', '.join(ts_features)}")
                if model.preprocessing_metadata.get('percentage_columns'):
                    pct_features = model.preprocessing_metadata['percentage_columns']
                    preprocessing_info.append(f"✅ {len(pct_features)} 個百分比欄位已轉換為數值：{', '.join(pct_features)}")
                if model.preprocessing_metadata.get('categorical_features'):
                    cat_features = model.preprocessing_metadata['categorical_features']
                    preprocessing_info.append(f"✅ {len(cat_features)} 個類別型特徵已進行獨熱編碼：{', '.join(cat_features)}")
                if preprocessing_info:
                    st.info("\n".join(preprocessing_info))
            else:
                model = GradientDescentModel(
                    loss=loss_function,
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    tol=tol,
                    use_scaling=use_scaling
                )
                model.fit(X_train, y_train, record_history=True, categorical_features=categorical_features)
                
                # 顯示預處理資訊
                preprocessing_info = []
                if model.preprocessing_metadata.get('thousand_separator_columns'):
                    ts_features = model.preprocessing_metadata['thousand_separator_columns']
                    preprocessing_info.append(f"✅ {len(ts_features)} 個千分位分隔符欄位已轉換為數值：{', '.join(ts_features)}")
                if model.preprocessing_metadata.get('percentage_columns'):
                    pct_features = model.preprocessing_metadata['percentage_columns']
                    preprocessing_info.append(f"✅ {len(pct_features)} 個百分比欄位已轉換為數值：{', '.join(pct_features)}")
                if model.preprocessing_metadata.get('categorical_features'):
                    cat_features = model.preprocessing_metadata['categorical_features']
                    preprocessing_info.append(f"✅ {len(cat_features)} 個類別型特徵已進行獨熱編碼：{', '.join(cat_features)}")
                if use_scaling:
                    preprocessing_info.append("✅ 數值型特徵已進行標準化")
                if preprocessing_info:
                    st.info("\n".join(preprocessing_info))
            
            # 儲存模型到 session state
            st.session_state['trained_model'] = model
            st.session_state['training_X'] = X_train
            st.session_state['training_y'] = y_train
            st.session_state['test_X'] = X_test
            st.session_state['test_y'] = y_test
            st.session_state['split_data'] = split_data
            
            st.success("✅ 模型訓練完成！")
            
        except Exception as e:
            st.error(f"❌ 訓練失敗：{str(e)}")
            st.stop()

# ========== 步驟 9: 訓練結果顯示 ==========
if 'trained_model' in st.session_state and st.session_state['trained_model'] is not None:
    st.markdown("---")
    st.markdown("### 步驟 9: 訓練結果")
    
    model = st.session_state['trained_model']
    X_train = st.session_state['training_X']
    y_train = st.session_state['training_y']
    split_data = st.session_state.get('split_data', False)
    X_test = st.session_state.get('test_X', None)
    y_test = st.session_state.get('test_y', None)
    
    # 取得模型資訊
    model_info = model.get_info()
    
    # 顯示模型基本資訊
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("模型類型", model_info['model_name'])
    with col2:
        st.metric("目標變數數量", len(model_info['target_names']))
    with col3:
        st.metric("特徵數量", model_info['n_features'])
    with col4:
        if split_data:
            st.metric("資料分割", "是")
        else:
            st.metric("資料分割", "否")
    with col5:
        # 顯示正則化資訊（僅線性回歸）
        if isinstance(model, LinearRegressionModel) and model_info.get('regularization'):
            reg_type = model_info['regularization']
            if reg_type == 'l1':
                reg_display = "L1 (Lasso)"
            elif reg_type == 'l2':
                reg_display = "L2 (Ridge)"
            else:
                reg_display = "無"
            st.metric("正則化", reg_display)
        else:
            st.metric("正則化", "無")
    
    # 顯示正則化強度（如果有）
    if isinstance(model, LinearRegressionModel) and model_info.get('regularization') and model_info.get('alpha'):
        st.info(f"📌 **正則化設定**：{model_info['regularization'].upper()} 正則化，強度 alpha = {model_info['alpha']:.3f}")
    
    # 損失曲線（僅梯度下降）
    if isinstance(model, GradientDescentModel) and model.training_history:
        st.markdown("#### 📊 訓練損失曲線")
        loss_fig = plot_training_loss(model.training_history, model.loss)
        st.plotly_chart(loss_fig, use_container_width=True)
    
    # 模型參數（係數）
    st.markdown("#### 📈 模型參數（係數）")
    
    # 嘗試顯示帶 p 值的係數表格（僅線性回歸且單一目標變數）
    show_pvalues = (
        isinstance(model, LinearRegressionModel) and 
        not model_info.get('is_multi_output', False) and
        X_train is not None and y_train is not None
    )
    
    if show_pvalues:
        try:
            # 重新應用預處理以獲取預處理後的資料（與模型訓練時一致）
            # 這樣才能正確計算 p 值和進行診斷
            X_processed = None
            if hasattr(model, 'preprocessing_metadata') and hasattr(model, 'encoder'):
                from utils.data_preprocessor import preprocess_features
                # 重新應用預處理（使用 fit=False，因為模型已經訓練過）
                X_processed, _, _, _ = preprocess_features(
                    X_train,
                    categorical_features=model.preprocessing_metadata.get('categorical_features', []),
                    use_scaling=False,  # 線性回歸不使用標準化
                    fit=False,
                    scaler=None,
                    encoder=model.encoder
                )
                X_train_values = X_processed.values
            else:
                # 如果沒有預處理元資料，使用原始資料
                X_train_values = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            
            y_train_values = y_train.values if isinstance(y_train, pd.DataFrame) else y_train
            
            # 確保資料是數值型且沒有無限值或 NaN
            X_train_values = np.array(X_train_values, dtype=np.float64)
            y_train_values = np.array(y_train_values, dtype=np.float64).flatten()
            
            # 檢查是否有無效值
            if np.any(np.isnan(X_train_values)) or np.any(np.isinf(X_train_values)):
                raise ValueError("X 資料包含 NaN 或 Inf 值")
            if np.any(np.isnan(y_train_values)) or np.any(np.isinf(y_train_values)):
                raise ValueError("y 資料包含 NaN 或 Inf 值")
            
            params_df = display_model_parameters_with_pvalues(
                model_info, 
                X_train_values, 
                y_train_values
            )
            
            # 檢查是否有 p 值計算成功
            p_value_col = 'p 值' if 'p 值' in params_df.columns else None
            if p_value_col:
                valid_p_count = params_df[p_value_col].apply(lambda x: x != 'N/A' and not pd.isna(x) if isinstance(x, str) else not pd.isna(x)).sum()
                total_count = len(params_df)
                
                if valid_p_count == 0:
                    # 執行診斷以提供更具體的信息
                    from utils.visualizer import diagnose_pvalue_issues
                    
                    # 獲取特徵名稱 - 優先使用預處理後資料的列名（最準確）
                    if X_processed is not None and isinstance(X_processed, pd.DataFrame):
                        feature_names = list(X_processed.columns)
                    else:
                        # 如果沒有預處理後的 DataFrame，使用 model_info 中的特徵名稱
                        feature_names = model_info.get('feature_names', [])
                        
                        # 如果特徵名稱不匹配，嘗試從預處理元資料構建
                        if not feature_names or len(feature_names) != X_train_values.shape[1]:
                            if hasattr(model, 'preprocessing_metadata'):
                                metadata = model.preprocessing_metadata
                                numeric_names = metadata.get('numeric_features', [])
                                encoded_names = metadata.get('encoded_feature_names', [])
                                
                                # 構建特徵名稱列表：數值特徵 + 編碼後的類別特徵
                                feature_names = numeric_names + encoded_names
                    
                    # 如果仍然不匹配，使用默認名稱
                    if not feature_names or len(feature_names) != X_train_values.shape[1]:
                        feature_names = None  # 讓診斷函數使用默認命名
                    
                    diagnostics = diagnose_pvalue_issues(X_train_values, y_train_values, feature_names)
                    
                    warning_msg = "⚠️ **所有 p 值都無法計算**\n\n"
                    warning_msg += "**診斷結果：**\n"
                    warning_msg += f"- 樣本數：{diagnostics['n_samples']}\n"
                    warning_msg += f"- 特徵數：{diagnostics['n_features']}\n"
                    warning_msg += f"- 樣本/特徵比：{diagnostics['sample_feature_ratio']:.2f}\n"
                    warning_msg += f"- 自由度：{diagnostics['degrees_of_freedom']}\n"
                    
                    if diagnostics['condition_number'] is not None:
                        warning_msg += f"- 條件數：{diagnostics['condition_number']:.2e}\n"
                    
                    if diagnostics['issues']:
                        warning_msg += "\n**發現的問題：**\n"
                        for issue in diagnostics['issues']:
                            warning_msg += f"- {issue}\n"
                    
                    # 顯示具體的問題特徵
                    if diagnostics['constant_feature_names']:
                        warning_msg += f"\n**常數特徵列表：**\n"
                        for name in diagnostics['constant_feature_names']:
                            warning_msg += f"- `{name}`\n"
                    
                    if diagnostics['highly_correlated_pairs']:
                        warning_msg += f"\n**高度相關的特徵對（|r| > 0.95）：**\n"
                        for pair in diagnostics['highly_correlated_pairs'][:10]:  # 最多顯示10對
                            corr_val = pair['correlation']
                            warning_msg += f"- `{pair['feature1']}` 與 `{pair['feature2']}` (r = {corr_val:.4f})\n"
                        if len(diagnostics['highly_correlated_pairs']) > 10:
                            warning_msg += f"- ... 還有 {len(diagnostics['highly_correlated_pairs']) - 10} 對高度相關的特徵\n"
                    
                    warning_msg += "\n**建議解決方案：**\n"
                    if diagnostics['n_samples'] <= diagnostics['n_features']:
                        warning_msg += "- **增加樣本數量或減少特徵數量**\n"
                        warning_msg += "  - 建議樣本數至少是特徵數的 3-5 倍\n"
                    if diagnostics['has_constant_features']:
                        warning_msg += "- **立即移除常數特徵**：這些特徵對模型沒有貢獻\n"
                        if diagnostics['constant_feature_names']:
                            warning_msg += f"  - 需要移除的特徵：{', '.join([f'`{name}`' for name in diagnostics['constant_feature_names']])}\n"
                    if diagnostics['has_multicollinearity']:
                        warning_msg += "- **處理多重共線性**：\n"
                        if diagnostics['highly_correlated_pairs']:
                            warning_msg += "  - 從高度相關的特徵對中移除其中一個特徵\n"
                        warning_msg += "  - 或使用主成分分析（PCA）降維\n"
                        warning_msg += "  - 或考慮使用正則化方法（Ridge 或 Lasso 回歸）\n"
                    if diagnostics['condition_number'] is None or diagnostics['condition_number'] > 1e12:
                        warning_msg += "- **檢查完全線性相關**：某些特徵可能是其他特徵的線性組合\n"
                    
                    st.warning(warning_msg)
                elif valid_p_count < total_count:
                    st.warning(f"⚠️ 部分 p 值無法計算（{valid_p_count}/{total_count} 個成功）。可能原因：\n"
                              "- 某些特徵存在完全線性相關\n"
                              "- 標準誤差為 0 或接近 0")
            
            # 顯示帶 p 值的係數表格
            st.dataframe(params_df, use_container_width=True)
            
            # p 值說明
            st.info(
                "💡 **p 值說明**：\n\n"
                "- **p 值**：檢驗係數是否顯著不為零的統計指標\n"
                "- **顯著性標記**：\n"
                "  • `***`：p < 0.001（極顯著）\n"
                "  • `**`：p < 0.01（非常顯著）\n"
                "  • `*`：p < 0.05（顯著）\n"
                "  • `.`：p < 0.1（邊緣顯著）\n"
                "  • 空白：p ≥ 0.1（不顯著）\n\n"
                "- **解讀**：p 值越小，表示該特徵對目標變數的影響越顯著\n"
                "- **一般標準**：p < 0.05 通常認為該特徵對目標變數有顯著影響\n\n"
                "- **N/A 說明**：如果 p 值顯示為 N/A，可能是因為：\n"
                "  • 特徵之間存在完全線性相關（多重共線性）\n"
                "  • 樣本數量太少或特徵數量太多\n"
                "  • 矩陣不可逆，無法計算標準誤差"
            )
        except Exception as e:
            # 如果計算 p 值失敗，回退到普通顯示
            params_df = display_model_parameters(model_info)
            st.dataframe(params_df, use_container_width=True)
            st.warning(f"⚠️ 無法計算 p 值：{str(e)}\n\n"
                      "可能原因：\n"
                      "- 特徵之間存在完全線性相關（多重共線性）\n"
                      "- 樣本數量太少或特徵數量太多\n"
                      "- 資料包含無效值（NaN 或 Inf）")
    else:
        params_df = display_model_parameters(model_info)
        st.dataframe(params_df, use_container_width=True)
    
    # 添加係數解釋說明
    categorical_features_list = model.preprocessing_metadata.get('categorical_features', [])
    has_categorical = len(categorical_features_list) > 0
    
    if len(target_variables) == 1:
        explanation = (
            "💡 **係數解釋**：\n\n"
            "**數值型特徵係數**：\n"
            "- **係數值**：表示該特徵對目標變數的影響程度\n"
            "- **正係數**：特徵值增加 1 單位時，目標變數會增加（係數值）單位\n"
            "- **負係數**：特徵值增加 1 單位時，目標變數會減少（係數絕對值）單位\n"
            "- **係數絕對值大小**：表示影響的強度\n\n"
        )
        
        if has_categorical:
            explanation += (
                "**類別型特徵係數**（經過獨熱編碼）：\n"
                "- 類別特徵經過獨熱編碼後，每個類別值變成一個二進制特徵（0 或 1）\n"
                "- **係數值**：表示相對於基準類別（第一個類別），該類別對目標變數的影響\n"
                "- **正係數**：該類別會使目標變數增加（係數值）單位，相對於基準類別\n"
                "- **負係數**：該類別會使目標變數減少（係數絕對值）單位，相對於基準類別\n"
                "- **範例**：如果「顏色」特徵有「紅、藍、綠」三個值，且「紅」是基準類別：\n"
                "  • 顏色_藍的係數 = 0.5 → 相對於紅色，藍色會使目標變數增加 0.5\n"
                "  • 顏色_綠的係數 = -0.3 → 相對於紅色，綠色會使目標變數減少 0.3\n\n"
            )
        
        explanation += (
            "**截距**：當所有數值型特徵為 0，且所有類別型特徵為基準類別時的預測值"
        )
        
        st.info(explanation)
    else:
        explanation = (
            "💡 **多目標變數係數解釋**：\n\n"
            "- 每個目標變數都有獨立的係數和截距\n\n"
            "**數值型特徵係數**：\n"
            "- **係數值**：表示該特徵對對應目標變數的影響程度\n"
            "- **正係數**：特徵值增加時，目標變數會增加\n"
            "- **負係數**：特徵值增加時，目標變數會減少\n\n"
        )
        
        if has_categorical:
            explanation += (
                "**類別型特徵係數**（經過獨熱編碼）：\n"
                "- 類別特徵經過獨熱編碼後，每個類別值變成一個二進制特徵\n"
                "- **係數值**：表示相對於基準類別（第一個類別），該類別對目標變數的影響\n"
                "- 每個目標變數都有獨立的類別係數\n\n"
            )
        
        explanation += (
            "**截距**：當所有特徵為 0 時該目標變數的預測值"
        )
        
        st.info(explanation)
    
    # 如果有類別特徵，顯示哪些特徵是類別特徵編碼後的結果
    if has_categorical:
        st.markdown("##### 📋 類別型特徵說明")
        encoded_feature_names = model.preprocessing_metadata.get('encoded_feature_names', [])
        
        # 嘗試從 encoder 獲取基準類別資訊
        encoder = model.encoder if hasattr(model, 'encoder') else None
        base_categories = {}
        
        if encoder is not None and hasattr(encoder, 'categories_'):
            for i, orig_feat in enumerate(categorical_features_list):
                if i < len(encoder.categories_):
                    categories = encoder.categories_[i]
                    if len(categories) > 0:
                        base_categories[orig_feat] = str(categories[0])
        
        # 按原始類別特徵分組
        categorical_groups = {}
        for encoded_name in encoded_feature_names:
            # 編碼後的特徵名稱格式：原始特徵名_類別值
            for orig_feat in categorical_features_list:
                if encoded_name.startswith(f"{orig_feat}_"):
                    if orig_feat not in categorical_groups:
                        categorical_groups[orig_feat] = []
                    categorical_groups[orig_feat].append(encoded_name)
                    break
        
        for orig_feat, encoded_features in categorical_groups.items():
            base_cat = base_categories.get(orig_feat, "第一個類別值")
            st.write(f"**{orig_feat}**（類別型特徵）：")
            st.write(f"  - 編碼後的特徵：{', '.join(encoded_features)}")
            st.write(f"  - 基準類別：**{base_cat}**（在係數表格中不顯示，係數為 0）")
            st.write(f"  - 其他類別的係數表示相對於基準類別「{base_cat}」的影響")
            st.write("")
    
    # 顯示係數統計資訊（僅單一目標變數時）
    if not model_info.get('is_multi_output', False) and len(params_df) > 1:
        st.markdown("##### 📊 係數統計資訊")
        coef_values = []
        for idx, row in params_df.iterrows():
            if row['特徵'] != '截距':
                try:
                    coef_val = float(row['係數'])
                    coef_values.append(coef_val)
                except:
                    pass
        
        if coef_values:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("最大係數", f"{max(coef_values, key=abs):.6f}")
            with col2:
                st.metric("最小係數", f"{min(coef_values, key=abs):.6f}")
            with col3:
                st.metric("平均絕對值", f"{np.mean([abs(x) for x in coef_values]):.6f}")
            with col4:
                st.metric("係數總數", len(coef_values))
            
            # 顯示影響最大的前 5 個特徵
            coef_with_features = []
            coef_idx = 0
            for idx, row in params_df.iterrows():
                if row['特徵'] != '截距':
                    try:
                        coef_val = float(row['係數'])
                        coef_with_features.append((abs(coef_val), coef_val, row['特徵']))
                    except:
                        pass
            
            coef_with_features.sort(reverse=True, key=lambda x: x[0])
            top_features = coef_with_features[:min(5, len(coef_with_features))]
            
            if top_features:
                st.markdown("##### 🔝 影響最大的前 5 個特徵（按係數絕對值）")
                top_df = pd.DataFrame({
                    '排名': range(1, len(top_features) + 1),
                    '特徵': [feat for _, _, feat in top_features],
                    '係數值': [f"{coef_val:.6f}" for _, coef_val, _ in top_features],
                    '係數絕對值': [f"{abs_val:.6f}" for abs_val, _, _ in top_features]
                })
                st.dataframe(top_df, use_container_width=True)
    
    # 評估指標
    if split_data and X_test is not None and y_test is not None:
        # 如果有測試集，顯示訓練集和測試集的評估指標
        st.markdown("#### 📊 評估指標")
        
        # 訓練集評估（始終使用單次評估）
        st.markdown("##### 🎯 訓練集評估指標")
        y_train_pred = model.predict(X_train)
        train_metrics_df = display_evaluation_metrics(y_train, y_train_pred)
        st.dataframe(train_metrics_df, use_container_width=True)
        
        # 測試集評估（根據選擇的評估方式）
        st.markdown("##### 🎯 測試集評估指標")
        
        if evaluation_method == "repeated":
            # 重複隨機分割評估
            from utils.evaluator import repeated_random_split_evaluate, format_metrics_with_std
            
            # 準備模型參數
            model_params = {}
            if algorithm == "線性回歸":
                model_class = LinearRegressionModel
                # 添加正則化參數（如果有的話）
                regularization = None
                if regularization_type == "L1 (Lasso)":
                    regularization = 'l1'
                elif regularization_type == "L2 (Ridge)":
                    regularization = 'l2'
                if regularization:
                    model_params = {
                        'regularization': regularization,
                        'alpha': alpha
                    }
            else:  # 梯度下降
                model_class = GradientDescentModel
                model_params = {
                    'loss': loss_function,
                    'learning_rate': learning_rate,
                    'max_iter': max_iter,
                    'tol': tol,
                    'use_scaling': use_scaling
                }
            
            # 執行重複隨機分割評估
            with st.spinner(f"正在執行重複隨機分割評估（{n_repeats} 次）..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(current, total):
                    progress_bar.progress(current / total)
                    status_text.text(f"進度：{current}/{total} 次評估完成")
                
                try:
                    # 使用原始資料（未擴增、未分割）進行評估
                    # 注意：這裡需要使用原始 X 和 y，而不是擴增後的
                    # 如果 X 和 y 不在作用域內，使用 X_train + X_test 和 y_train + y_test 合併
                    if 'X' not in locals() or 'y' not in locals():
                        # 如果 X 和 y 不在作用域內，從訓練集和測試集合併
                        if X_test is not None and y_test is not None:
                            X_eval = pd.concat([X_train, X_test], ignore_index=True)
                            y_eval = pd.concat([y_train, y_test], ignore_index=True)
                        else:
                            # 如果沒有測試集，使用訓練集
                            X_eval = X_train.copy()
                            y_eval = y_train.copy()
                    else:
                        # 使用原始 X 和 y
                        X_eval = X.copy()
                        y_eval = y.copy()
                    
                    # 取得類別型特徵列表（如果存在）
                    categorical_features_for_eval = st.session_state.get('categorical_features', None)
                    if categorical_features_for_eval:
                        categorical_features_for_eval = [f for f in categorical_features_for_eval if f in X_eval.columns]
                    
                    # 準備擴增參數（如果啟用擴增）
                    aug_params = None
                    if enable_augmentation:
                        aug_params = {
                            'noise_type': augmentation_params['noise_type'],
                            'noise_strength': augmentation_params['noise_strength'],
                            'multiplier': augmentation_params['multiplier']
                        }
                    
                    mean_metrics, std_metrics, all_results = repeated_random_split_evaluate(
                        model_class,
                        model_params,
                        X_eval,  # 使用原始資料
                        y_eval,  # 使用原始資料
                        test_size=test_size,
                        n_repeats=n_repeats,
                        base_random_state=42,
                        progress_callback=progress_callback,
                        categorical_features=categorical_features_for_eval,
                        augmentation_params=aug_params  # 傳遞擴增參數
                    )
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # 顯示評估結果（平均值 ± 標準差）
                    st.success(f"✅ 重複隨機分割評估完成（{n_repeats} 次）")
                    
                    # 格式化並顯示結果
                    results_df = format_metrics_with_std(mean_metrics, std_metrics)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # 顯示 R² 比較
                    from sklearn.metrics import r2_score
                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2_mean = mean_metrics['R²']
                    test_r2_std = std_metrics['R²']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("訓練集 R²", f"{train_r2:.4f}")
                    with col2:
                        st.metric("測試集 R²（平均值 ± 標準差）", f"{test_r2_mean:.4f} ± {test_r2_std:.4f}")
                    
                    # 顯示所有評估結果（預設摺疊）
                    with st.expander(f"📋 查看所有評估結果（{n_repeats} 次評估的詳細記錄）", expanded=False):
                        st.markdown("**每次評估的詳細結果：**")
                        detailed_df = pd.DataFrame(all_results)
                        # 格式化數值顯示
                        for col in detailed_df.columns:
                            if col not in ['重複次數', '隨機種子', '訓練集樣本數', '測試集樣本數']:
                                detailed_df[col] = detailed_df[col].apply(lambda x: f"{x:.6f}")
                        st.dataframe(detailed_df, use_container_width=True)
                        
                        # 顯示統計摘要
                        st.markdown("**統計摘要：**")
                        summary_data = {
                            '指標': ['訓練集 R²', '測試集 R²', '訓練集 MSE', '測試集 MSE', '訓練集 MAE', '測試集 MAE'],
                            '平均值': [
                                np.mean([r['訓練集 R²'] for r in all_results]),
                                np.mean([r['測試集 R²'] for r in all_results]),
                                np.mean([r['訓練集 MSE'] for r in all_results]),
                                np.mean([r['測試集 MSE'] for r in all_results]),
                                np.mean([r['訓練集 MAE'] for r in all_results]),
                                np.mean([r['測試集 MAE'] for r in all_results])
                            ],
                            '標準差': [
                                np.std([r['訓練集 R²'] for r in all_results]),
                                np.std([r['測試集 R²'] for r in all_results]),
                                np.std([r['訓練集 MSE'] for r in all_results]),
                                np.std([r['測試集 MSE'] for r in all_results]),
                                np.std([r['訓練集 MAE'] for r in all_results]),
                                np.std([r['測試集 MAE'] for r in all_results])
                            ],
                            '最小值': [
                                np.min([r['訓練集 R²'] for r in all_results]),
                                np.min([r['測試集 R²'] for r in all_results]),
                                np.min([r['訓練集 MSE'] for r in all_results]),
                                np.min([r['測試集 MSE'] for r in all_results]),
                                np.min([r['訓練集 MAE'] for r in all_results]),
                                np.min([r['測試集 MAE'] for r in all_results])
                            ],
                            '最大值': [
                                np.max([r['訓練集 R²'] for r in all_results]),
                                np.max([r['測試集 R²'] for r in all_results]),
                                np.max([r['訓練集 MSE'] for r in all_results]),
                                np.max([r['測試集 MSE'] for r in all_results]),
                                np.max([r['訓練集 MAE'] for r in all_results]),
                                np.max([r['測試集 MAE'] for r in all_results])
                            ]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        # 格式化數值顯示
                        for col in ['平均值', '標準差', '最小值', '最大值']:
                            summary_df[col] = summary_df[col].apply(lambda x: f"{x:.6f}")
                        st.dataframe(summary_df, use_container_width=True)
                    
                    # 如果測試集 R² 明顯低於訓練集 R²，提示可能過擬合
                    if test_r2_mean < train_r2 - 0.1:
                        st.warning("⚠️ 測試集 R² 明顯低於訓練集 R²，可能存在過擬合（Overfitting）問題。建議：減少特徵數量、增加訓練資料、或使用正則化。")
                
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ 重複隨機分割評估失敗：{str(e)}\n\n將使用單次評估結果。")
                    # 回退到單次評估
                    y_test_pred = model.predict(X_test)
                    test_metrics_df = display_evaluation_metrics(y_test, y_test_pred)
                    st.dataframe(test_metrics_df, use_container_width=True)
                    
                    from sklearn.metrics import r2_score
                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("訓練集 R²", f"{train_r2:.4f}")
                    with col2:
                        st.metric("測試集 R²", f"{test_r2:.4f}")
                    
                    if test_r2 < train_r2 - 0.1:
                        st.warning("⚠️ 測試集 R² 明顯低於訓練集 R²，可能存在過擬合（Overfitting）問題。建議：減少特徵數量、增加訓練資料、或使用正則化。")
        else:
            # 單次評估（原有邏輯）
            y_test_pred = model.predict(X_test)
            test_metrics_df = display_evaluation_metrics(y_test, y_test_pred)
            st.dataframe(test_metrics_df, use_container_width=True)
            
            # 比較訓練集和測試集的 R²
            from sklearn.metrics import r2_score
            
            # 記錄單次評估結果
            single_eval_result = {
                '評估類型': '單次評估',
                '訓練集樣本數': len(X_train),
                '測試集樣本數': len(X_test),
                '隨機種子': 42 if split_data else None
            }
            
            # 多目標變數時，計算每個目標的 R² 和平均 R²
            if len(target_variables) > 1:
                train_r2_scores = [r2_score(y_train.iloc[:, i], y_train_pred.iloc[:, i]) 
                                  for i in range(len(target_variables))]
                test_r2_scores = [r2_score(y_test.iloc[:, i], y_test_pred.iloc[:, i]) 
                                 for i in range(len(target_variables))]
                train_r2 = np.mean(train_r2_scores)
                test_r2 = np.mean(test_r2_scores)
                
                # 顯示每個目標變數的 R²
                st.markdown("##### 📊 各目標變數 R² 比較")
                r2_comparison_data = {
                    '目標變數': target_variables,
                    '訓練集 R²': [f"{score:.4f}" for score in train_r2_scores],
                    '測試集 R²': [f"{score:.4f}" for score in test_r2_scores]
                }
                r2_comparison_df = pd.DataFrame(r2_comparison_data)
                st.dataframe(r2_comparison_df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("訓練集平均 R²", f"{train_r2:.4f}")
                with col2:
                    st.metric("測試集平均 R²", f"{test_r2:.4f}")
                
                # 記錄評估結果
                single_eval_result.update({
                    '訓練集 R²': train_r2,
                    '測試集 R²': test_r2
                })
            else:
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("訓練集 R²", f"{train_r2:.4f}")
                with col2:
                    st.metric("測試集 R²", f"{test_r2:.4f}")
                
                # 記錄評估結果
                single_eval_result.update({
                    '訓練集 R²': train_r2,
                    '測試集 R²': test_r2
                })
            
            # 計算並記錄其他指標
            from utils.evaluator import evaluate_model
            train_metrics = evaluate_model(y_train, y_train_pred)
            test_metrics = evaluate_model(y_test, y_test_pred)
            single_eval_result.update({
                '訓練集 MSE': train_metrics['MSE'],
                '測試集 MSE': test_metrics['MSE'],
                '訓練集 MAE': train_metrics['MAE'],
                '測試集 MAE': test_metrics['MAE']
            })
            
            # 顯示所有評估結果（預設摺疊）
            with st.expander("📋 查看完整評估記錄", expanded=False):
                st.markdown("**評估結果記錄：**")
                eval_record_df = pd.DataFrame([single_eval_result])
                # 格式化數值顯示
                for col in eval_record_df.columns:
                    if col not in ['評估類型', '訓練集樣本數', '測試集樣本數'] and eval_record_df[col].dtype in [np.float64, np.float32]:
                        eval_record_df[col] = eval_record_df[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
                st.dataframe(eval_record_df, use_container_width=True)
                
                st.markdown("**詳細指標：**")
                detailed_metrics = pd.DataFrame({
                    '指標': ['R²', 'MSE', 'MAE'],
                    '訓練集': [
                        f"{train_metrics['R²']:.6f}",
                        f"{train_metrics['MSE']:.6f}",
                        f"{train_metrics['MAE']:.6f}"
                    ],
                    '測試集': [
                        f"{test_metrics['R²']:.6f}",
                        f"{test_metrics['MSE']:.6f}",
                        f"{test_metrics['MAE']:.6f}"
                    ]
                })
                st.dataframe(detailed_metrics, use_container_width=True)
                
            # 如果測試集 R² 明顯低於訓練集 R²，提示可能過擬合
            if test_r2 < train_r2 - 0.1:
                st.warning("⚠️ 測試集 R² 明顯低於訓練集 R²，可能存在過擬合（Overfitting）問題。建議：減少特徵數量、增加訓練資料、或使用正則化。")
    else:
        # 沒有測試集，只顯示訓練集評估指標
        st.markdown("#### 📊 評估指標（訓練集）")
        y_pred = model.predict(X_train)
        metrics_df = display_evaluation_metrics(y_train, y_pred)
        st.dataframe(metrics_df, use_container_width=True)
        
        # 記錄評估結果
        from utils.evaluator import evaluate_model
        train_metrics = evaluate_model(y_train, y_pred)
        
        # 顯示所有評估結果（預設摺疊）
        with st.expander("📋 查看完整評估記錄", expanded=False):
            st.markdown("**評估結果記錄：**")
            eval_record = {
                '評估類型': '訓練集評估（無測試集）',
                '訓練集樣本數': len(X_train),
                '訓練集 R²': train_metrics['R²'],
                '訓練集 MSE': train_metrics['MSE'],
                '訓練集 MAE': train_metrics['MAE']
            }
            eval_record_df = pd.DataFrame([eval_record])
            # 格式化數值顯示
            for col in eval_record_df.columns:
                if col not in ['評估類型', '訓練集樣本數']:
                    eval_record_df[col] = eval_record_df[col].apply(lambda x: f"{x:.6f}")
            st.dataframe(eval_record_df, use_container_width=True)
            
            st.markdown("**詳細指標：**")
            detailed_metrics = pd.DataFrame({
                '指標': ['R²', 'MSE', 'MAE'],
                '訓練集': [
                    f"{train_metrics['R²']:.6f}",
                    f"{train_metrics['MSE']:.6f}",
                    f"{train_metrics['MAE']:.6f}"
                ]
            })
            st.dataframe(detailed_metrics, use_container_width=True)
    
    # 模型保存
    st.markdown("---")
    st.markdown("### 步驟 10: 保存模型")
    
    # 根據目標變數數量生成預設模型名稱
    if len(target_variables) == 1:
        default_model_name = f"{algorithm}_{target_variables[0]}"
    else:
        default_model_name = f"{algorithm}_{len(target_variables)}targets"
    
    model_name = st.text_input(
        "模型名稱",
        value=default_model_name,
        help="輸入模型名稱（不含副檔名）"
    )
    
    if st.button("💾 保存模型", use_container_width=True):
        if model_name:
            try:
                filepath = save_model(model, model_name)
                st.success(f"✅ 模型已保存至：{filepath}")
            except Exception as e:
                st.error(f"❌ 保存失敗：{str(e)}")
        else:
            st.warning("請輸入模型名稱")
