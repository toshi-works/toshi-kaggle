import numpy as np
from sklearn.model_selection import KFold


def train_lgb_cv_and_predict(X_train, y_train, df_test, random_state=42):
    """5-Fold交差検証を行い、あわせてテストデータの予測値を計算する関数"""

    # 1. 現在の訓練データから実際に存在するcategory列を自動抽出
    current_cat_cols = (
        X_train.select_dtypes(include=["category"]).columns.tolist()
    )

    # 2. テストデータの型同期と列の厳選（X_trainと100%一致させる）
    test_df = df_test.copy()
    for col in current_cat_cols:
        if col in test_df.columns:
            test_df[col] = pd.Categorical(
                test_df[col],
                categories=X_train[col].cat.categories,
                ordered=X_train[col].cat.ordered,
            )
    X_test_selected = test_df[X_train.columns]

    # 3. 受け皿の準備
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test_selected))

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # 4. 5-Fold ループ
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # モデル定義と学習
        model = lgb.LGBMRegressor(random_state=random_state)
        model.fit(X_tr, y_tr, categorical_feature=current_cat_cols)

        # 予測の蓄積
        oof_preds[val_idx] = model.predict(X_va)
        test_preds += model.predict(X_test_selected) / 5

    # 5. 総合スコア（CV）の算出
    cv_score = root_mean_squared_error(y_train, oof_preds)
    print("\n" + "=" * 40)
    print(f"🏆 5-Fold 交差検証全体のCVスコア(RMSE): {cv_score:.2f}")
    print("=" * 40)

    # テストデータの予測結果を返す
    return test_preds