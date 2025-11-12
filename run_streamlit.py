"""
Streamlit 應用程式啟動腳本（用於 PyInstaller 打包）

此腳本用於在打包後的環境中啟動 Streamlit 應用程式。
"""
import sys
import os
from pathlib import Path

# 明確導入 streamlit，讓 PyInstaller 能夠檢測到
# 這必須在頂部，這樣 PyInstaller 才能正確打包
import streamlit
import streamlit.web.cli

# 確保在正確的目錄中執行
if getattr(sys, 'frozen', False):
    # 如果是打包後的執行檔
    application_path = Path(sys.executable).parent
else:
    # 如果是開發環境
    application_path = Path(__file__).parent

os.chdir(application_path)

# 啟動 Streamlit
if __name__ == "__main__":
    # 如果是打包後的執行檔，確保模組路徑正確
    if getattr(sys, 'frozen', False):
        # 添加 _internal 目錄到路徑（PyInstaller 的模組目錄）
        base_path = Path(sys.executable).parent
        internal_path = base_path / '_internal'
        if internal_path.exists():
            sys.path.insert(0, str(internal_path))
        # 也添加當前目錄
        sys.path.insert(0, str(base_path))
    
    # 使用已經在頂部導入的 streamlit.web.cli
    stcli = streamlit.web.cli
    
    # 設定 Streamlit 參數
    sys.argv = [
        "streamlit",
        "run",
        str(application_path / "app.py"),
        "--server.port=8501",
        "--server.address=localhost",
        "--browser.gatherUsageStats=false"
    ]
    
    sys.exit(stcli.main())

