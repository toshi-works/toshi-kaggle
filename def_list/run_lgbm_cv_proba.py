from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd

#1番最初のセルで実行して「ストック」しておく
def run_lgbm_cv_proba(X, y):
    """
    ヒント(X)と答え(y)を渡すだけで、自動で型変換から
    5分割クロスバリデーション（確率予測）までを行う関数
    """
    # 文字列の列を自動で一括 category 型に
    X_copy = X.copy() # 元のデータを壊さないための安全マージン
    obj_cols = X_copy.select_dtypes(include=['object']).columns
    X_copy[obj_cols] = X_copy[obj_cols].astype('category')
    
    # 2. 5分割のクロスバリデーションの準備
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_lgb_proba = np.zeros(len(X_copy))
    
    print("--- 🤖 LightGBM クロスバリデーション（確率版）開始 ---")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_copy, y)):
        X_train, y_train = X_copy.iloc[train_idx], y[train_idx]
        X_val, y_val = X_copy.iloc[val_idx], y[val_idx]
        
        model = LGBMClassifier(random_state=42, n_estimators=100, learning_rate=0.05)
        model.fit(X_train, y_train)
        
        # 確率（1である確率）を予測して箱に保存
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        oof_preds_lgb_proba[val_idx] = y_pred_proba
        
        # 各Foldの手元スコア確認
        score = accuracy_score(y[val_idx], np.where(y_pred_proba > 0.5, 1, 0))
        print(f"Fold {fold+1} の正解率: {score:.5f}")
        
    # 🌟【重要！】全体のガチの実力スコアを表示
    total_score = accuracy_score(y, np.where(oof_preds_lgb_proba > 0.5, 1, 0))
    print(f"✨ 全体（OOF）の最終CVスコア: {total_score:.5f}")
        
    # 🌟【ここを追加！】計算した確率の配列を、関数の外に呼び出し元へ戻す
    return oof_preds_lgb_proba