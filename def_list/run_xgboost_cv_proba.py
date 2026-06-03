from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

# 👇 これをストック用として1番最初のセルで実行しておく
def run_xgboost_cv_proba(X, y):
    """
    ヒント(X)と答え(y)を渡すだけで、XGBoost特有の型変換を行い、
    5分割クロスバリデーション（確率予測）までを行う関数
    """
    X_copy = X.copy() # 元のデータを壊さないための安全マージン
    
    # 1. XGBoost用に、一回 str にしてから category 型に変換（インデント修正）
    for col in X_copy.select_dtypes(include=['object', 'category']).columns:
        X_copy[col] = X_copy[col].astype(str).astype('category')
        
    # 2. 5分割のクロスバリデーションの準備（関数内で定義）
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_xgb_proba = np.zeros(len(X_copy))
    
    print("--- 🔥 XGBoost クロスバリデーション（確率版）開始 ---")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_copy, y)):
        X_train, y_train = X_copy.iloc[train_idx], y[train_idx]
        X_val, y_val = X_copy.iloc[val_idx], y[val_idx]
        
        # enable_categorical=True を忘れずセット
        model_xgb = XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.05, enable_categorical=True)
        model_xgb.fit(X_train, y_train)
        
        # 確率で予測を出して箱に保存
        y_pred_proba_xgb = model_xgb.predict_proba(X_val)[:, 1]
        oof_preds_xgb_proba[val_idx] = y_pred_proba_xgb
        
    # 3. 全部のFoldが終わった後に、全体のスコアを計算
    xgb_actual_score = accuracy_score(y, np.where(oof_preds_xgb_proba > 0.5, 1, 0))
    print(f"XGBoostのCVスコア: {xgb_actual_score:.5f}")
    
    # 🌟 計算した確率の配列を外に戻す
    return oof_preds_xgb_proba