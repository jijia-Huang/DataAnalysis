## Context
這是一個全新的專案，需要建立 Streamlit Web 應用程式的基礎架構。應用程式將提供三個主要頁面：資料分析、訓練、預測。目標使用者是非程式人員，因此介面設計必須直觀易用。

## Goals / Non-Goals
- **Goals**:
  - 建立可擴展的 Streamlit 應用架構
  - 實作多頁面導航系統
  - 建立資料分析頁面的核心功能（CSV 上傳、資料分布視覺化）
  - 確保程式碼結構清晰、模組化
  - 提供良好的錯誤處理和使用者回饋

- **Non-Goals**:
  - 訓練頁面和預測頁面的實作（後續提案）
  - 模型訓練功能（後續提案）
  - 應用程式打包（後續提案）

## Decisions
- **頁面架構**：使用 Streamlit 的 `st.pages` 功能（Streamlit 1.28+）或 `st.sidebar` + 條件渲染實現多頁面導航
  - **理由**：`st.pages` 是 Streamlit 推薦的多頁面方式，但需要較新版本；如果版本不支援，使用 sidebar 導航作為備選方案
  - **實作**：優先嘗試 `st.pages`，如果不支援則使用 sidebar 導航

- **資料視覺化**：使用 Plotly 進行互動式圖表
  - **理由**：與 Streamlit 整合良好，提供互動式體驗，適合非程式人員使用

- **檔案結構**：
  ```
  app.py                 # 主應用檔案
  pages/
    data_analysis.py    # 資料分析頁面
    training.py          # 訓練頁面（預留）
    prediction.py        # 預測頁面（預留）
  utils/
    data_loader.py       # 資料載入工具
    visualizer.py        # 視覺化工具
  models/                # 模型儲存目錄（預留）
  data/                  # 資料目錄（已存在）
  ```

- **錯誤處理**：使用 try-except 包裝關鍵操作，並使用 `st.error()` 顯示友善的錯誤訊息
  - **理由**：非程式人員需要清晰易懂的錯誤提示

- **資料編碼處理**：自動偵測 CSV 編碼（UTF-8, BIG5），優先嘗試 UTF-8
  - **理由**：專案約束中提到資料檔案可能有編碼問題

## Risks / Trade-offs
- **Streamlit 版本相容性**：`st.pages` 需要 Streamlit >= 1.28
  - **緩解**：在 requirements.txt 中明確指定版本，或使用 sidebar 導航作為備選

- **大型檔案處理**：大型 CSV 檔案可能導致記憶體問題
  - **緩解**：使用 Streamlit 的 `@st.cache_data` 快取，並考慮限制檔案大小或提供分頁顯示

- **資料格式驗證**：不同格式的 CSV 可能導致解析失敗
  - **緩解**：提供清晰的錯誤訊息，並嘗試多種分隔符號（逗號、分號、Tab）

## Migration Plan
這是初始建立，無需遷移。

## Open Questions
- Streamlit 版本選擇：使用最新穩定版還是特定版本？
  - **決策**：使用 `streamlit>=1.28.0` 以支援 `st.pages`，如果遇到問題再降級

