"""
資料分析頁面

提供 CSV 檔案上傳、資料預覽、統計摘要和視覺化功能。
"""

import streamlit as st
import pandas as pd
from utils.data_loader import load_csv_file, validate_dataframe, get_data_info
from utils.visualizer import (
    create_histograms_for_numeric_columns,
    create_scatter_matrix,
    create_statistics_summary
)


# Streamlit 會自動將此檔案識別為頁面
# 檔案名稱中的 "1_" 表示順序，"📊" 是圖示，"資料分析" 是頁面標題

st.title("📊 資料分析")
st.markdown("---")

# 說明文字
st.info(
    "👋 歡迎使用資料分析功能！\n\n"
    "請上傳一個 CSV 檔案，系統會自動分析資料並顯示：\n"
    "- 資料預覽\n"
    "- 統計摘要\n"
    "- 資料分布圖表\n"
    "- 變數關係圖表"
)

# 檔案上傳
uploaded_file = st.file_uploader(
    "選擇 CSV 檔案",
    type=['csv'],
    help="請選擇一個 CSV 格式的資料檔案"
)

if uploaded_file is not None:
    # 顯示載入中訊息
    with st.spinner("正在載入資料..."):
        # 載入資料
        df, error_message = load_csv_file(uploaded_file)
        
        if error_message:
            st.error(f"❌ {error_message}")
            st.stop()
        
        # 驗證資料
        is_valid, validation_error = validate_dataframe(df)
        
        if not is_valid:
            st.error(f"❌ {validation_error}")
            st.stop()
        
        # 將資料存入 session state
        st.session_state['data_analysis_df'] = df
        
        st.success("✅ 資料載入成功！")
    
    # 顯示資料資訊
    data_info = get_data_info(df)
    st.markdown("### 📋 資料資訊")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("資料筆數", f"{data_info['shape'][0]:,}")
    with col2:
        st.metric("欄位數量", data_info['shape'][1])
    with col3:
        st.metric("記憶體使用", f"{data_info['memory_usage'] / 1024:.2f} KB")
    
    # 顯示缺失值警告
    missing_values = {k: v for k, v in data_info['missing_values'].items() if v > 0}
    if missing_values:
        st.warning(
            f"⚠️ 發現缺失值：\n" +
            "\n".join([f"- {col}: {count} 個" for col, count in missing_values.items()])
        )
    
    st.markdown("---")
    
    # 資料預覽
    st.markdown("### 👀 資料預覽")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"顯示前 10 筆資料（總共 {len(df):,} 筆）")
    
    st.markdown("---")
    
    # 統計摘要
    st.markdown("### 📈 統計摘要")
    stats_df = create_statistics_summary(df)
    st.dataframe(stats_df, use_container_width=True)
    
    st.markdown("---")
    
    # 資料分布視覺化
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if numeric_columns:
        st.markdown("### 📊 資料分布圖表")
        
        # 為每個數值欄位建立直方圖
        histograms = create_histograms_for_numeric_columns(df)
        
        if histograms:
            # 使用 tabs 來組織多個圖表
            tabs = st.tabs([f"📊 {col}" for col in numeric_columns[:6]])  # 最多顯示 6 個
            
            for i, (tab, fig) in enumerate(zip(tabs, histograms[:6])):
                with tab:
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 變數關係視覺化
        if len(numeric_columns) >= 2:
            st.markdown("### 🔗 變數關係圖表")
            scatter_matrix = create_scatter_matrix(df)
            
            if scatter_matrix:
                st.plotly_chart(scatter_matrix, use_container_width=True)
            else:
                st.info("無法建立散點圖矩陣。資料可能過大或格式不適合。")
    else:
        st.info("📝 資料中沒有數值欄位，無法顯示分布圖表。")

else:
    # 沒有上傳檔案時的提示
    st.info("👆 請在上方選擇一個 CSV 檔案開始分析")

