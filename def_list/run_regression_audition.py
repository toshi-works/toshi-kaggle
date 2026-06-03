import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error

# 👇 これを1番最初のセルで実行して「ストック」しておく
def run_regression_audition(X, y, test_size=0.2, random_state=42):
    """
    生のヒント(X)と答え(y)を渡すだけで、
    前処理(標準化)から3つのモデルの一括評価までを自動で行う関数
    """
    # 1. データの分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # 2. パイプラインの定義（完全固定の定型文）
    pipelines = {
        'LinearRegression（線形回帰）': Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())]),
        'RandomForest（ランダムフォレスト）': Pipeline([('scaler', StandardScaler()), ('rf', RandomForestRegressor(random_state=random_state))]),
        'SVR（サポートベクターマシン）': Pipeline([('scaler', StandardScaler()), ('svr', SVR())])
    }
    
    print("--- パイプライン版・一括計測 ---")
    results = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        error = root_mean_squared_error(y_val, y_pred)
        print(f"{name} の平均誤差: {error:.4f}")
        results[name] = pipe # 後から使えるように学習済みのモデルも保存しておく
        
    return results