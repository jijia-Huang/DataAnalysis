"""
DataAnalysis Platform - 主應用程式

互動式資料分析與機器學習訓練平台

注意：Streamlit 會自動偵測 pages/ 目錄下的頁面檔案並建立導航。
如果使用較舊版本的 Streamlit，請使用 sidebar 導航方式。
"""

import streamlit as st

# 設定頁面配置
st.set_page_config(
    page_title="DataAnalysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主頁面內容
st.title("📊 DataAnalysis Platform")
st.markdown("---")
st.markdown(
    "歡迎使用 DataAnalysis Platform！\n\n"
    "這是一個互動式資料分析與機器學習訓練平台，提供以下功能：\n"
    "- 📊 **資料分析**：上傳 CSV 檔案並查看資料分布\n"
    "- 🎯 **模型訓練**：訓練線性回歸模型（即將推出）\n"
    "- 🔮 **模型預測**：使用訓練好的模型進行預測（即將推出）\n\n"
    "請使用左側導航選單選擇功能頁面。"
)

# 注意：Streamlit 會自動偵測 pages/ 目錄下的檔案並建立導航
# 頁面檔案命名格式：數字_圖示_標題.py（例如：1_📊_資料分析.py）
# Streamlit 會自動在側邊欄建立導航選單

