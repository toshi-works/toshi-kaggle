

# 「いらない列」の条件（例：空欄が半分以上ある列、など）を指定して自動で削る、、目的変数は減らす
select_train = train.drop(columns=["Id"])

# LightGBMを行う前の文字列の一括返還
cat_cols = select_train.select_dtypes(include=["object"]).columns.tolist()
# LightGBMが読み込めるように、文字列の列を「category型」に一発変換
type_dict = {col: "category" for col in cat_cols}
select_train = select_train.astype(type_dict)

# 1. 訓練データを使って、エリア（Neighborhood）ごとの平均価格を計算する（カンペ作り）
neighborhood_mean = select_train.groupby("Neighborhood")["SalePrice"].mean()

# 2. 訓練データにも、テストデータ（test_df）にも、そのカンペの数字を割り振る！
select_train["Target_Encoded_Neighborhood"] = select_train["Neighborhood"].map(
    neighborhood_mean
)
test["Target_Encoded_Neighborhood"] = test["Neighborhood"].map(
    neighborhood_mean)

# LightGBMのデフォルト値のシンプルコード
import lightgbm as lgb
from sklearn.model_selection import train_test_split
# ==========================================
# 問題用紙（X）と正解（y）に分ける
# ==========================================
# 「SalePrice（住宅価格）」だけを削ったものが（X）
X = select_train.drop(columns=["SalePrice"])
y = select_train["SalePrice"]

# データを「8割」と「2割」に分割
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# ==========================================
# LightGBMをデフォルト値で作成して学習
# ==========================================
model = lgb.LGBMRegressor(random_state=42)
# 学習スタート（前処理で作った cat_cols をここで渡します）
model.fit(X_train, y_train, categorical_feature=cat_cols)

# ==========================================
# 模擬試験を解いて、点数（RMSE）を見る
# ==========================================
from sklearn.metrics import root_mean_squared_error
# 模擬試験（X_val）を解かせる
val_pred = model.predict(X_val)
# 平均二乗誤差（RMSE）を計算して表示
rmse_score = root_mean_squared_error(y_val, val_pred)
print("-----------------------------------------")
print(f"RMSEスコア: {rmse_score:.2f}")
print("-----------------------------------------")

import matplotlib.pyplot as plt
import numpy as np
# LightGBMに重要度を計算してもらう
importance = model.feature_importances_
feature_names = X.columns

# 見やすくするためにデータフレームにまとめる
df_importance = pd.DataFrame(
    {"Feature": feature_names, "Importance": importance}
).sort_values("Importance", ascending=False)

# ➔ 上位15個を画面にプリント
print("🏆 LightGBMが選んだ重要変数トップ15:")
print(df_importance.head(15).to_string(index=False))

# 📊 グラフ（横棒グラフ）で視覚的に表示する
plt.figure(figsize=(10, 6))
plt.barh(
    df_importance["Feature"].head(15)[::-1],
    df_importance["Importance"].head(15)[::-1],
)
plt.xlabel("Importance (Split Count)")
plt.title("Top 15 Most Important Features")
plt.show()

# ==========================================
# 1. 削る基準（しきい値）を決める
# ==========================================
# 💡 重要度（Importance）の点数が「10点未満」の、ほぼ役に立っていない変数をあぶり出します
threshold = 10
drop_features = df_importance[df_importance["Importance"] < threshold][
    "Feature"
].tolist()

print(f"🗑️ 今回ノイズとして削る変数（{len(drop_features)}個）:")
print(drop_features)

# ==========================================
# 2. 新しい問題用紙（変数を削ったデータ）を作る
# ==========================================
# 元の問題用紙（X）から、役に立っていない変数だけを削除（drop）します
X_selected = X.drop(columns=drop_features)

# 新しいカテゴリ変数のリストを更新（削られた変数をリストから除外する）
cat_cols_selected = [col for col in cat_cols if col in X_selected.columns]

# データを「練習用（8割）」と「手元の答え合わせ用（2割）」に再分割
# 💡 random_state=42 を同じにすることで、前回と「全く同じ問題」で対照実験ができます
X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# ==========================================
# 3. 厳選したメンバーで、LightGBMを再学習
# ==========================================
model_s = lgb.LGBMRegressor(random_state=42)
model_s.fit(X_train_s, y_train_s, categorical_feature=cat_cols_selected)

# 模擬試験を解かせて、新しいスコアを計算
val_pred_s = model_s.predict(X_val_s)
new_rmse_score = root_mean_squared_error(y_val_s, val_pred_s)

# ==========================================
# 4. 前回のスコアとの運命の答え合わせ
# ==========================================
print("\n=========================================")
print(f"📊 前回（27個すべて）のRMSE: {rmse_score:.2f}")
print(f"🔥 今回（ノイズ削減後）のRMSE: {new_rmse_score:.2f}")
print("=========================================")

# 結果の判定
if new_rmse_score < rmse_score:
    print("🎉 スコアが下がりました！ノイズカット大成功です！精度が上がっています。")
elif new_rmse_score > rmse_score:
    print(
        "🤔 あれ、少しスコアが上がって（悪化して）しまいましたね。AIにとっては、一見無駄に見えても必要なヒントだったのかもしれません。"
    )
else:
    print("😐 スコアは全く変わりませんでした。")

test["Target_Encoded_Neighborhood"] = test["Neighborhood"].map(
    neighborhood_mean)

# ========================================================
# 💡 【安全版】手前で加えたカラムの影響を完全にリセットして同期する
# ========================================================

# 1. 現在の X（訓練データ）にある列の中で、本当に category 型になっている列だけを最新のリストにする
current_cat_cols = X.select_dtypes(include=["category"]).columns.tolist()

# 2. その「今、実際に存在する category 列」だけで安全にループを回す
for col in current_cat_cols:
    # 念のため、テストデータ側にもその列が存在する場合だけ処理する
    if col in test.columns:
        test[col] = pd.Categorical(
            test[col],
            categories=X[col].cat.categories,
            ordered=X[col].cat.ordered,
        )

print("✅ テストデータの型同期が安全に完了しました！")
print(f"同期された列: {current_cat_cols}")

# 💡 テストデータの列を、X（訓練データ）と「全く同じ列、全く同じ並び順」に強制的に切り出す
test_selected = test[X.columns]

#=======================================================
# 1. テストデータの型を安全に同期する ========================================================
current_cat_cols = X.select_dtypes(include=["category"]).columns.tolist()

for col in current_cat_cols:
    if col in test.columns:
        test[col] = pd.Categorical(
            test[col],
            categories=X[col].cat.categories,
            ordered=X[col].cat.ordered,
        )

# ========================================================
# 💡 【今回の特効薬】テストデータの列を X と 100% 完全に一致させる
# ========================================================
# これにより、testの中に余計な「SalePrice」や「Id」が残っていても、きれいに削ぎ落とされます
test_selected = test[X.columns]


# ==========================================
# 2. 準備：OOF（器）とテスト予測の受け皿を作る
# ==========================================
num_train = len(X)
num_test = len(test_selected)  # 👈 修正後のデータを使用

oof_preds = np.zeros(num_train)
test_preds = np.zeros(num_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================
# 3. 運命の5回ループ
# ==========================================
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"🎬 --- Fold {fold + 1} / 5 の学習を開始 ---")

    X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y.iloc[val_idx]

    model_f = lgb.LGBMRegressor(random_state=42)
    model_f.fit(X_train_f, y_train_f, categorical_feature=current_cat_cols)
# *CatBoost/LightGBM以外は、categorical_feature=current_cat_colsを消し、カテゴリーデータは、ダミー編透かしてからループを回す
    oof_preds[val_idx] = model_f.predict(X_val_f)

    # 💡 100% 訓練データと同じ列数になった「test_selected」を渡して予測！
    test_preds += model_f.predict(test_selected) / 5


# ==========================================
# 4. 総合スコア（CV）の表示
# ==========================================
cv_score = root_mean_squared_error(y, oof_preds)
print("\n=========================================")
print(f"🏆 5-Fold 交差検証（OOF）全体のRMSE: {cv_score:.2f}")
print("=========================================")