## Why
線性回歸模型在處理高維資料或特徵數量較多時，容易出現過擬合問題。添加懲罰項（正則化）可以幫助控制模型複雜度，提高泛化能力。使用者需要能夠選擇是否使用正則化，以及選擇正則化類型（L1/L2）和強度。

## What Changes
- 在線性回歸訓練時添加懲罰項選項
- 支援 L1 正則化（Lasso）和 L2 正則化（Ridge）
- 允許使用者設定正則化強度（alpha 參數）
- 更新訓練頁面 UI，添加正則化配置選項
- 添加可展開/收起的說明區塊，解釋 L1 和 L2 正則化的作用和差異
- 修改 LinearRegressionModel 以支援正則化參數

## Impact
- Affected specs: training-page
- Affected code: 
  - `models/linear_regression.py` - 添加正則化支援
  - `pages/2_🎯_訓練.py` - 添加正則化配置 UI

