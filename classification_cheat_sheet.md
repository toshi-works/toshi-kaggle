import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# 分類用の評価指標をインポート
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 代表的な分類モデルのインポート
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
# 必要に応じて xgboost から XGBClassifier なども追加可能

# 1. サンプルデータの準備（乳がんの良性・悪性分類データ）
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 
# ※stratify=y を入れることで、訓練とテストで正解ラベルの比率を均等にします（分類の鉄則です）

# 2. 比較したい分類モデルを辞書形式で定義
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "LightGBM (Balanced)": LGBMClassifier(random_state=42, is_unbalance=True, verbose=-1)
}

# 3. ループで一括学習・評価
results = []

for name, model in models.items():
    # 学習
    model.fit(X_train, y_train)
    
    # 予測（通常のラベル予測と、ROC-AUC用の確率予測）
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # 「1」になる確率を抽出
    
    # 評価指標の計算
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    # 結果の保存
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

# 4. 結果をデータフレームにして比較（F1スコアが高い順にソート）
df_results = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
print(df_results)