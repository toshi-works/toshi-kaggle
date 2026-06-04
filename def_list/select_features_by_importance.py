import lightgbm as lgb
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split


def select_features_by_importance(df, target_col, cat_cols, threshold=10):
    """LightGBMの重要度をベースに、ノイズとなる特徴量を削除する関数"""
    # 1. データの用意
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. 初回学習（デフォルト値）
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_train, y_train, categorical_feature=cat_cols)

    # 初回スコアの計算
    val_pred = model.predict(X_val)
    base_rmse = root_mean_squared_error(y_val, val_pred)

    # 3. 重要度の低い特徴量をあぶり出す
    df_importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": model.feature_importances_}
    )
    drop_features = df_importance[df_importance["Importance"] < threshold][
        "Feature"
    ].tolist()

    # 4. 特徴量の削除と再分割
    X_selected = X.drop(columns=drop_features)
    cat_cols_selected = [col for col in cat_cols if col in X_selected.columns]

    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    # 5. 厳選メンバーで再学習
    model_s = lgb.LGBMRegressor(random_state=42)
    model_s.fit(X_train_s, y_train_s, categorical_feature=cat_cols_selected)

    new_rmse = root_mean_squared_error(y_val_s, model_s.predict(X_val_s))

    # 結果のスコア比較表示
    print("\n" + "=" * 40)
    print(f"📊 削減前 RMSE: {base_rmse:.2f}")
    print(f"🔥 削減後 RMSE: {new_rmse:.2f} (ノイズ削除: {len(drop_features)}個)")
    print("=" * 40)

    # 厳選された特徴量データ、正解ラベル、最新のカテゴリ列リストを返す
    return X_selected, y, cat_cols_selected