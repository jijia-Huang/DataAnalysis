"""
預測頁面

提供模型預測功能，支援單筆資料輸入和 CSV 檔案批次預測。
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from typing import Optional, Dict, Any

from utils.model_manager import load_model, list_models, load_field_info
from utils.data_loader import load_csv_file, validate_dataframe
from utils.prediction_validator import validate_prediction_input
from models.base_model import BaseModel
from utils.data_preprocessor import preprocess_features


def calculate_manual_prediction(model: BaseModel, input_df: pd.DataFrame, target_name: Optional[str] = None) -> Dict[str, Any]:
    """
    手動計算預測值（用於驗算）
    
    Args:
        model: 訓練好的模型
        input_df: 輸入資料（DataFrame）
        target_name: 目標變數名稱（如果是多輸出模型）
    
    Returns:
        dict: 包含驗算過程的字典
    """
    # 獲取模型資訊
    model_info = model.get_info()
    
    # 檢查是否為線性回歸模型（只有線性回歸可以手動驗算）
    if model_info.get('model_name') not in ['Linear Regression', 'Gradient Descent']:
        return {
            'can_calculate': False,
            'message': '此模型類型不支援手動驗算'
        }
    
    # 預處理輸入資料（與預測時相同）
    if hasattr(model, 'preprocessing_metadata'):
        preprocessing_metadata = model.preprocessing_metadata
        categorical_features = preprocessing_metadata.get('categorical_features', [])
        use_scaling = getattr(model, 'use_scaling', False) if hasattr(model, 'use_scaling') else False
        scaler = getattr(model, 'scaler', None)
        encoder = getattr(model, 'encoder', None)
    else:
        categorical_features = []
        use_scaling = False
        scaler = None
        encoder = None
    
    X_processed, _, _, _ = preprocess_features(
        input_df,
        categorical_features=categorical_features,
        use_scaling=use_scaling,
        fit=False,
        scaler=scaler,
        encoder=encoder
    )
    
    # 確保特徵順序與訓練時一致
    if model.feature_names is not None:
        X_processed = X_processed[model.feature_names]
    
    # 獲取係數和截距
    is_multi_output = model_info.get('is_multi_output', False)
    
    if is_multi_output:
        coefficients_list = model_info.get('coefficients', [])
        intercepts_list = model_info.get('intercepts', [])
        target_names = model_info.get('target_names', [])
        
        if target_name is None:
            target_name = target_names[0] if target_names else None
        
        if target_name and target_name in target_names:
            target_idx = target_names.index(target_name)
            coefficients = coefficients_list[target_idx] if target_idx < len(coefficients_list) else []
            intercept = intercepts_list[target_idx] if target_idx < len(intercepts_list) else 0
        else:
            return {
                'can_calculate': False,
                'message': f'找不到目標變數：{target_name}'
            }
    else:
        coefficients = model_info.get('coefficients', [])
        intercept = model_info.get('intercept', 0)
        target_name = model_info.get('target_names', [None])[0]
    
    # 手動計算預測值
    feature_values = X_processed.iloc[0].values
    manual_prediction = intercept + np.dot(coefficients, feature_values)
    
    # 建立驗算過程
    calculation_steps = []
    calculation_steps.append(f"**截距** = {intercept:.6f}")
    
    for i, (feature_name, feature_value) in enumerate(zip(model.feature_names, feature_values)):
        coef = coefficients[i] if i < len(coefficients) else 0
        product = coef * feature_value
        calculation_steps.append(f"**{feature_name}** × {coef:.6f} = {feature_value:.6f} × {coef:.6f} = {product:.6f}")
    
    calculation_steps.append(f"**總和** = {manual_prediction:.6f}")
    
    return {
        'can_calculate': True,
        'target_name': target_name,
        'intercept': intercept,
        'coefficients': coefficients,
        'feature_names': model.feature_names,
        'feature_values': feature_values.tolist(),
        'manual_prediction': manual_prediction,
        'calculation_steps': calculation_steps
    }


st.title("🔮 模型預測")
st.markdown("---")

# 說明文字
st.info(
    "👋 歡迎使用模型預測功能！\n\n"
    "請按照以下步驟進行預測：\n"
    "1. 選擇要使用的模型（訓練頁面的模型或已保存的模型）\n"
    "2. 選擇預測方式（單筆資料或 CSV 批次預測）\n"
    "3. 輸入資料並執行預測\n"
    "4. 查看並匯出預測結果"
)

# ========== 步驟 1: 模型選擇 ==========
st.markdown("### 步驟 1: 選擇模型")

model: Optional[BaseModel] = None
model_source = None
model_filepath: Optional[str] = None  # 記錄模型檔案路徑（如果從檔案載入）

# 檢查 session state 中是否有訓練好的模型
if 'trained_model' in st.session_state and st.session_state['trained_model'] is not None:
    model = st.session_state['trained_model']
    model_source = "session"
    
    st.success("✅ 已從訓練頁面載入模型")
    
    # 顯示模型資訊
    model_info = model.get_info()
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**模型類型**：{model_info.get('model_name', 'Unknown')}")
        st.write(f"**目標變數**：{', '.join(model_info.get('target_names', []))}")
    with col2:
        st.write(f"**特徵數量**：{len(model_info.get('feature_names', []))}")
        st.write(f"**多目標**：{'是' if model_info.get('is_multi_output', False) else '否'}")
    
    use_session_model = st.checkbox("使用此模型", value=True, key="use_session_model")
    
    if not use_session_model:
        model = None
        model_source = None

# 如果沒有 session state 的模型，或使用者選擇不使用，則載入檔案
if model is None:
    model_source = "file"
    
    # 取得已保存的模型列表
    saved_models = list_models()
    
    if not saved_models:
        st.warning("⚠️ 沒有找到已保存的模型。請先在訓練頁面訓練並保存模型。")
        st.stop()
    
    # 顯示模型選擇器
    model_options = {}
    for m in saved_models:
        if 'error' in m:
            display_name = f"{m['filename']} (載入錯誤)"
        else:
            model_name = m.get('model_name', 'Unknown')
            target_names = ', '.join(m.get('target_names', []))
            modified_time = m.get('modified_time', '').strftime('%Y-%m-%d %H:%M:%S') if m.get('modified_time') else 'Unknown'
            display_name = f"{model_name} | 目標: {target_names} | 保存時間: {modified_time}"
        model_options[display_name] = m['filepath']
    
    selected_model_display = st.selectbox(
        "選擇已保存的模型",
        options=list(model_options.keys()),
        help="選擇要使用的模型檔案"
    )
    
    if selected_model_display:
        selected_filepath = model_options[selected_model_display]
        
        # 檢查是否有錯誤
        selected_model_info = next((m for m in saved_models if m['filepath'] == selected_filepath), None)
        if selected_model_info and 'error' in selected_model_info:
            st.error(f"❌ 無法載入模型：{selected_model_info['error']}")
            st.stop()
        
        # 載入模型
        try:
            with st.spinner("正在載入模型..."):
                model = load_model(selected_filepath)
                model_filepath = selected_filepath  # 記錄檔案路徑
            
            st.success("✅ 模型載入成功！")
            
            # 優先從 JSON 檔案載入欄位資訊（如果存在）
            field_info = load_field_info(selected_filepath)
            if field_info:
                # 使用 JSON 檔案中的資訊
                model_info = field_info
            else:
                # 如果沒有 JSON 檔案，從模型取得資訊
                model_info = model.get_info()
            
            # 顯示模型資訊
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**模型類型**：{model_info.get('model_name', 'Unknown')}")
                st.write(f"**目標變數**：{', '.join(model_info.get('target_names', []))}")
            with col2:
                st.write(f"**特徵數量**：{len(model_info.get('feature_names', []))}")
                st.write(f"**多目標**：{'是' if model_info.get('is_multi_output', False) else '否'}")
        except Exception as e:
            st.error(f"❌ 載入模型失敗：{str(e)}")
            st.stop()

if model is None:
    st.warning("👆 請先選擇模型")
    st.stop()

st.markdown("---")

# ========== 步驟 2: 選擇預測方式 ==========
st.markdown("### 步驟 2: 選擇預測方式")

prediction_mode = st.radio(
    "選擇預測方式",
    options=["單筆資料預測", "CSV 檔案批次預測"],
    help="單筆預測適合快速測試，批次預測適合處理大量資料"
)

st.markdown("---")

# ========== 步驟 3: 執行預測 ==========
if prediction_mode == "單筆資料預測":
    st.markdown("### 步驟 3: 輸入資料並預測")
    
    # 優先從 JSON 檔案載入欄位資訊（如果存在）
    field_info = None
    json_loaded = False
    
    if model_source == "file" and model_filepath:
        # 如果是從檔案載入的模型，嘗試從 JSON 檔案載入欄位資訊
        field_info = load_field_info(model_filepath)
        if field_info:
            json_loaded = True
            st.info(f"✅ 已從 JSON 檔案載入欄位資訊：{model_filepath.replace('.joblib', '.json')}")
    
    # 如果沒有 JSON 檔案或從 session state 載入，則從模型取得資訊
    if field_info is None:
        model_info = model.get_info()
        field_info = {
            'original_columns': model_info.get('original_columns', []),
            'categorical_features': model_info.get('categorical_features', []),
            'numeric_features': model_info.get('numeric_features', []),
            'target_names': model_info.get('target_names', []),
            'feature_names': model_info.get('feature_names', [])
        }
        if model_source == "file" and model_filepath:
            st.warning(f"⚠️ 未找到 JSON 檔案，從模型載入欄位資訊。預期路徑：{model_filepath.replace('.joblib', '.json')}")
    
    # 從欄位資訊中取得特徵類型資訊
    original_columns = field_info.get('original_columns', [])
    categorical_features = field_info.get('categorical_features', [])
    numeric_features = field_info.get('numeric_features', [])
    target_names = field_info.get('target_names', [])
    feature_names = field_info.get('feature_names', [])
    
    # 決定需要輸入的特徵（優先使用 original_columns，否則使用 feature_names）
    if original_columns:
        required_features = original_columns
    else:
        required_features = feature_names
    
    # 確保目標變數不會出現在特徵列表中（雖然理論上不應該出現）
    required_features = [f for f in required_features if f not in target_names]
    
    if not required_features:
        st.error("❌ 模型沒有特徵資訊，無法進行預測")
        st.stop()
    
    # 確保類別型特徵和數值型特徵都不包含目標變數
    categorical_features = [f for f in categorical_features if f not in target_names]
    numeric_features = [f for f in numeric_features if f not in target_names]
    
    # 取得類別型特徵的有效值（從 encoder 中取得）
    encoder = getattr(model, 'encoder', None)
    valid_categories = {}
    
    # 如果 encoder 存在，說明有類別型特徵被編碼過
    if encoder and hasattr(encoder, 'categories_') and categorical_features:
        # encoder.categories_ 的順序與訓練時傳入的 categorical_features 順序一致
        # 建立特徵名稱到索引的映射
        preprocessing_metadata = getattr(model, 'preprocessing_metadata', {})
        encoded_feature_names = preprocessing_metadata.get('encoded_feature_names', [])
        
        if encoded_feature_names:
            # 從 encoded_feature_names 建立映射（更可靠）
            # encoded_feature_names 的順序與 encoder.categories_ 的順序一致
            seen_features = {}
            for enc_feat in encoded_feature_names:
                underscore_pos = enc_feat.find('_')
                if underscore_pos > 0:
                    orig_feat = enc_feat[:underscore_pos]
                    if orig_feat not in seen_features and orig_feat in categorical_features:
                        # 這是該特徵的第一個編碼特徵，記錄索引
                        seen_features[orig_feat] = len(seen_features)
        
            # 使用映射來取得類別值
            for feature in categorical_features:
                if feature in seen_features:
                    encoder_idx = seen_features[feature]
                    if encoder_idx < len(encoder.categories_):
                        categories = encoder.categories_[encoder_idx]
                        valid_categories[feature] = [str(cat) for cat in categories]
        else:
            # 如果沒有 encoded_feature_names，嘗試順序匹配
            for i, feature in enumerate(categorical_features):
                if i < len(encoder.categories_):
                    categories = encoder.categories_[i]
                    valid_categories[feature] = [str(cat) for cat in categories]
    
    # 顯示調試資訊（可選，用於診斷）
    with st.expander("🔍 特徵類型診斷資訊（點擊展開）", expanded=False):
        st.write(f"**資料來源**：{'✅ JSON 檔案' if json_loaded else '⚠️ 模型物件'}")
        if model_source == "file" and model_filepath:
            json_path = model_filepath.replace('.joblib', '.json')
            st.write(f"**JSON 檔案路徑**：{json_path}")
            from pathlib import Path
            json_file = Path(json_path)
            if json_file.exists():
                st.write(f"**JSON 檔案存在**：✅ 是")
            else:
                st.write(f"**JSON 檔案存在**：❌ 否")
        st.write(f"**需要輸入的特徵**：{len(required_features)} 個")
        st.write(f"  - {', '.join(required_features)}")
        st.write(f"**類別型特徵**：{len(categorical_features)} 個")
        if categorical_features:
            st.write(f"  - {', '.join(categorical_features)}")
        else:
            st.write("  - 無")
        st.write(f"**有類別值的特徵**：{len(valid_categories)} 個")
        if valid_categories:
            for feat, vals in valid_categories.items():
                st.write(f"  - {feat}: {len(vals)} 個類別值")
        else:
            st.write("  - 無")
        st.write(f"**數值型特徵**：{len(numeric_features)} 個")
        if numeric_features:
            st.write(f"  - {', '.join(numeric_features)}")
        else:
            st.write("  - 無")
        st.write(f"**原始欄位（original_columns）**：{len(original_columns)} 個")
        if original_columns:
            st.write(f"  - {', '.join(original_columns)}")
        else:
            st.write("  - 無")
        st.write("**從 field_info 取得的完整資訊：**")
        st.json(field_info)
    
    # 動態生成輸入表單
    input_data = {}
    
    with st.form("single_prediction_form"):
        st.write("**請輸入各特徵值：**")
        
        # 分成兩欄顯示
        cols = st.columns(2)
        col_idx = 0
        
        for feature in required_features:
            with cols[col_idx % 2]:
                # 優先檢查 valid_categories（最可靠的指標，因為它來自 encoder）
                # 如果特徵在 valid_categories 中，說明它是類別型特徵
                if feature in valid_categories:
                    # 類別型特徵使用 selectbox
                    valid_values = valid_categories[feature]
                    if valid_values:
                        input_data[feature] = st.selectbox(
                            f"{feature} *",
                            options=valid_values,
                            help=f"類別型特徵，請選擇一個值"
                        )
                    else:
                        # 如果沒有有效值列表，使用 text_input（應該不會發生，但作為備選）
                        st.warning(f"⚠️ 特徵 '{feature}' 的類別值資訊不可用")
                        input_data[feature] = st.text_input(
                            f"{feature} *",
                            value="",
                            help=f"類別型特徵，請輸入值"
                        )
                elif feature in categorical_features:
                    # 如果不在 valid_categories 但在 categorical_features 中
                    # 可能是 encoder 資訊缺失，但仍應視為類別型
                    st.warning(f"⚠️ 特徵 '{feature}' 被識別為類別型，但類別值資訊不可用")
                    input_data[feature] = st.text_input(
                        f"{feature} *",
                        value="",
                        help=f"類別型特徵，請輸入值"
                    )
                else:
                    # 數值型特徵使用 number_input
                    input_data[feature] = st.number_input(
                        f"{feature} *",
                        value=0.0,
                        format="%.2f",
                        help=f"數值型特徵"
                    )
            
            col_idx += 1
        
        submitted = st.form_submit_button("執行預測", use_container_width=True)
        
        if submitted:
            # 清除舊的預測結果（修復刷新問題）
            if 'prediction_result' in st.session_state:
                del st.session_state['prediction_result']
            if 'prediction_error' in st.session_state:
                del st.session_state['prediction_error']
            if 'prediction_warning' in st.session_state:
                del st.session_state['prediction_warning']
            if 'prediction_calculation' in st.session_state:
                del st.session_state['prediction_calculation']
            
            # 清除所有驗算相關的 session state
            keys_to_remove = [key for key in st.session_state.keys() if key.startswith('calc_')]
            for key in keys_to_remove:
                del st.session_state[key]
            
            # 將輸入資料轉換為 DataFrame
            input_df = pd.DataFrame([input_data])
            
            # 驗證輸入資料
            is_valid, error_msg, validation_info = validate_prediction_input(input_df, model)
            
            if not is_valid:
                st.session_state['prediction_error'] = error_msg
                st.session_state['prediction_result'] = None
            else:
                # 檢查並排除目標變數（如果存在）
                target_names = model.target_names or []
                if target_names:
                    target_vars_in_input = [col for col in input_df.columns if col in target_names]
                    if target_vars_in_input:
                        st.session_state['prediction_warning'] = (
                            f"⚠️ 注意：輸入資料包含目標變數欄位（{', '.join(target_vars_in_input)}），"
                            f"這些欄位將被自動排除，不會用於預測。"
                        )
                        input_df = input_df.drop(columns=target_vars_in_input)
                    else:
                        st.session_state['prediction_warning'] = None
                
                # 執行預測
                try:
                    with st.spinner("正在執行預測..."):
                        predictions = model.predict(input_df)
                    
                    # 合併輸入和預測結果
                    result_df = pd.concat([input_df, predictions], axis=1)
                    
                    # 保存結果到 session state
                    st.session_state['prediction_result'] = {
                        'input_df': input_df,
                        'predictions': predictions,
                        'result_df': result_df
                    }
                    st.session_state['prediction_error'] = None
                    
                except Exception as e:
                    st.session_state['prediction_error'] = f"❌ 預測失敗：{str(e)}"
                    st.session_state['prediction_result'] = None
    
    # 在表單外部顯示預測結果
    if 'prediction_error' in st.session_state and st.session_state['prediction_error']:
        st.error(st.session_state['prediction_error'])
    
    if 'prediction_warning' in st.session_state and st.session_state['prediction_warning']:
        st.warning(st.session_state['prediction_warning'])
    
    if 'prediction_result' in st.session_state and st.session_state['prediction_result'] is not None:
        result = st.session_state['prediction_result']
        
        st.success("✅ 預測完成！")
        
        # 顯示預測結果
        st.markdown("### 預測結果")
        
        # 顯示輸入資料
        st.markdown("#### 輸入資料")
        st.dataframe(result['input_df'], use_container_width=True)
        
        # 顯示預測值
        st.markdown("#### 預測值")
        
        # 顯示預測值表格
        st.dataframe(result['predictions'], use_container_width=True)
        
        # 為每個預測值添加驗算功能
        for target_col in result['predictions'].columns:
            pred_value = result['predictions'][target_col].iloc[0]
            
            # 使用 expander 顯示驗算過程（預設展開）
            with st.expander(f"🔍 {target_col} 驗算過程（點擊展開/收起）", expanded=False):
                try:
                    calc_result = calculate_manual_prediction(model, result['input_df'], target_col)
                    
                    if calc_result.get('can_calculate', False):
                        st.markdown("##### 📐 計算公式")
                        st.code(f"預測值 = 截距 + Σ(係數 × 特徵值)")
                        
                        st.markdown("##### 📊 計算步驟")
                        for step in calc_result['calculation_steps']:
                            st.markdown(step)
                        
                        st.markdown("##### ✅ 驗證")
                        model_pred = pred_value
                        manual_pred = calc_result['manual_prediction']
                        diff = abs(model_pred - manual_pred)
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("模型預測值", f"{model_pred:.6f}")
                        with col_b:
                            st.metric("手動計算值", f"{manual_pred:.6f}")
                        with col_c:
                            st.metric("差異", f"{diff:.10f}")
                        
                        if diff < 1e-6:
                            st.success("✅ 驗算通過！模型預測值與手動計算值一致。")
                        else:
                            st.warning(f"⚠️ 差異較大（{diff:.10f}），可能由於數值精度或模型實現差異。")
                    else:
                        st.info(f"ℹ️ {calc_result.get('message', '無法進行驗算')}")
                except Exception as e:
                    st.error(f"❌ 驗算過程出錯：{str(e)}")
                    import traceback
                    with st.expander("查看詳細錯誤資訊"):
                        st.code(traceback.format_exc())
        
        # 提供下載功能
        csv = result['result_df'].to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下載預測結果 (CSV)",
            data=csv,
            file_name="prediction_result.csv",
            mime="text/csv"
        )

else:  # CSV 檔案批次預測
    st.markdown("### 步驟 3: 上傳 CSV 檔案並預測")
    
    uploaded_file = st.file_uploader(
        "上傳 CSV 檔案",
        type=['csv'],
        help="請上傳與訓練資料相同格式的 CSV 檔案"
    )
    
    if uploaded_file is not None:
        # 載入 CSV 檔案
        with st.spinner("正在載入 CSV 檔案..."):
            input_df, error_message = load_csv_file(uploaded_file)
        
        if error_message:
            st.error(f"❌ {error_message}")
            st.stop()
        
        if input_df is None:
            st.error("❌ 無法讀取檔案")
            st.stop()
        
        # 驗證 DataFrame
        is_valid, validation_error = validate_dataframe(input_df)
        if not is_valid:
            st.error(f"❌ {validation_error}")
            st.stop()
        
        st.success(f"✅ CSV 檔案載入成功！（{input_df.shape[0]} 筆，{input_df.shape[1]} 欄位）")
        
        # 顯示資料預覽
        st.markdown("#### 資料預覽")
        st.dataframe(input_df.head(10), use_container_width=True)
        
        # 驗證輸入資料
        st.markdown("#### 資料驗證")
        is_valid, error_msg, validation_info = validate_prediction_input(input_df, model)
        
        if not is_valid:
            st.error(f"❌ 資料驗證失敗：{error_msg}")
            if validation_info and 'valid_categories' in validation_info:
                st.info("💡 提示：請確認類別型特徵的值在有效值範圍內")
        else:
            # 檢查並排除目標變數（如果存在）
            target_names = model.target_names or []
            if target_names:
                target_vars_in_input = [col for col in input_df.columns if col in target_names]
                if target_vars_in_input:
                    st.warning(
                        f"⚠️ 注意：CSV 檔案包含目標變數欄位（{', '.join(target_vars_in_input)}），"
                        f"這些欄位將被自動排除，不會用於預測。"
                    )
                    input_df = input_df.drop(columns=target_vars_in_input)
            
            st.success("✅ 資料驗證通過！")
            
            # 執行批次預測
            if st.button("執行批次預測", use_container_width=True, type="primary"):
                try:
                    # 限制批次大小（避免記憶體問題）
                    max_batch_size = 10000
                    if len(input_df) > max_batch_size:
                        st.warning(f"⚠️ 資料量較大（{len(input_df)} 筆），將分批處理（每批 {max_batch_size} 筆）")
                        
                        all_predictions = []
                        progress_bar = st.progress(0)
                        
                        for i in range(0, len(input_df), max_batch_size):
                            batch_df = input_df.iloc[i:i+max_batch_size]
                            batch_predictions = model.predict(batch_df)
                            all_predictions.append(batch_predictions)
                            
                            progress = min((i + max_batch_size) / len(input_df), 1.0)
                            progress_bar.progress(progress)
                        
                        predictions = pd.concat(all_predictions, ignore_index=True)
                        progress_bar.empty()
                    else:
                        with st.spinner("正在執行批次預測..."):
                            predictions = model.predict(input_df)
                    
                    st.success(f"✅ 批次預測完成！（共 {len(predictions)} 筆）")
                    
                    # 顯示預測結果
                    st.markdown("### 預測結果")
                    
                    # 合併原始資料和預測結果
                    result_df = pd.concat([input_df.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
                    
                    # 顯示結果表格
                    st.dataframe(result_df, use_container_width=True, height=400)
                    
                    # 顯示統計資訊
                    st.markdown("#### 預測結果統計")
                    for target in predictions.columns:
                        st.write(f"**{target}**：")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("平均值", f"{predictions[target].mean():.4f}")
                        with col2:
                            st.metric("標準差", f"{predictions[target].std():.4f}")
                        with col3:
                            st.metric("最小值", f"{predictions[target].min():.4f}")
                        st.metric("最大值", f"{predictions[target].max():.4f}")
                    
                    # 提供下載功能
                    csv = result_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 下載預測結果 (CSV)",
                        data=csv,
                        file_name="batch_prediction_results.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ 預測失敗：{str(e)}")
                    import traceback
                    with st.expander("查看詳細錯誤資訊"):
                        st.code(traceback.format_exc())
