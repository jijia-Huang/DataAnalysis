## Context
應用程式開發完成後，需要打包交付給企劃團隊進行測試和評估。企劃團隊通常不具備 Python 開發環境，需要簡單易用的打包版本。

## Goals / Non-Goals
- **Goals**:
  - 提供獨立可執行檔，無需安裝 Python 環境
  - 提供完整的使用者手冊和安裝說明
  - 確保打包版本功能完整且穩定
  - 提供多種打包方式以適應不同需求

- **Non-Goals**:
  - 不提供自動更新功能（未來可擴展）
  - 不提供安裝程式（未來可擴展）
  - 不提供程式碼簽名（未來可擴展）

## Decisions
- **Decision**: 優先使用 PyInstaller 打包 Windows 版本
  - **Rationale**: Windows 是最常用的平台，PyInstaller 是最成熟的 Python 打包工具
  - **Alternatives considered**: 
    - cx_Freeze：功能較少，社群支援較弱
    - Nuitka：編譯時間長，相容性問題較多
    - py2exe：僅支援 Windows，功能有限

- **Decision**: 使用 `--onedir` 模式而非 `--onefile`
  - **Rationale**: `--onedir` 模式啟動速度更快，檔案結構更清晰
  - **Alternatives considered**: 
    - `--onefile`：單一檔案，但啟動速度慢，檔案較大

- **Decision**: 提供 Docker 容器作為可選方案
  - **Rationale**: Docker 提供跨平台支援和環境一致性
  - **Alternatives considered**: 
    - 僅提供可執行檔：限制跨平台使用
    - 僅提供 Docker：需要安裝 Docker，對非技術人員較複雜

- **Decision**: 撰寫完整的使用者手冊
  - **Rationale**: 企劃團隊需要詳細的使用說明
  - **Alternatives considered**: 
    - 僅提供基本說明：可能導致使用困難

## Implementation Details
- **PyInstaller 配置**:
  - 使用 `build.spec` 檔案進行配置
  - 設定隱藏導入：`streamlit`, `plotly`, `sklearn` 等
  - 設定資料檔案：必要的靜態資源
  - 設定應用程式資訊：名稱、版本、圖示等

- **打包流程**:
  1. 準備乾淨的虛擬環境
  2. 安裝所有依賴
  3. 執行 PyInstaller 打包
  4. 測試打包結果
  5. 在乾淨環境中驗證

- **文檔結構**:
  - `INSTALL.md`：安裝說明
  - `QUICKSTART.md`：快速開始指南
  - `USER_GUIDE.md`：使用者手冊
  - `FAQ.md`：常見問題
  - `TROUBLESHOOTING.md`：故障排除指南

- **測試策略**:
  - 功能測試：確保所有功能正常
  - 效能測試：確保啟動速度和記憶體使用合理
  - 跨平台測試：確保在不同環境中正常運作

## Risks / Trade-offs
- **Risk**: PyInstaller 打包可能遺漏某些依賴
  - **Mitigation**: 使用 `--collect-all` 選項，完整測試所有功能

- **Risk**: 打包檔案過大
  - **Mitigation**: 使用 `--onedir` 模式，壓縮分發

- **Risk**: 啟動速度慢
  - **Mitigation**: 使用 `--onedir` 模式，優化導入

- **Trade-off**: 單一檔案 vs 目錄結構
  - **Choice**: 選擇目錄結構（`--onedir`），因為啟動速度更快

## Open Questions
- 是否需要提供自動更新功能？
- 是否需要程式碼簽名（Code Signing）？
- 是否需要建立安裝程式（Installer）？

