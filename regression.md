import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 代表的なモデルのインポート
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
# 必要に応じて xgboost から XGBRegressor なども追加可能

# 1. サンプルデータの準備
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 比較したいモデルを辞書形式で定義
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "LightGBM": LGBMRegressor(random_state=42, verbose=-1) # verbose=-1で警告を非表示
}

# 3. ループで一括学習・評価
results = []

for name, model in models.items():
    # 学習
    model.fit(X_train, y_train)
    # 予測
    y_pred = model.predict(X_test)
    
    # 評価指標の計算
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    # 結果の保存
    results.append({
        "Model": name,
        "RMSE": rmse,
        "R2 Score": r2
    })

# 4. 結果をデータフレームにして比較
df_results = pd.DataFrame(results).sort_values(by="RMSE", ascending=True)
print(df_results)