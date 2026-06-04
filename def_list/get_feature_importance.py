import pandas as pd


def get_feature_importance(model, feature_names, top_n=10):
    """モデルから特徴量の重要度を抽出し、上位N個を並び替えて表示・出力する関数"""

    # 1. モデルから重要度を取り出してデータフレームにまとめる
    feature_imp_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": model.feature_importances_}
    )

    # 2. 重要度が高い順（降順）に並び替える
    feature_imp_df = feature_imp_df.sort_values(
        by="Importance", ascending=False
    ).reset_index(drop=True)

    # 3. 画面に見やすく表示する
    print(f"🏆 LightGBM 特徴量の重要度 トップ {top_n}:")
    print(feature_imp_df.head(top_n).to_string(index=True))
    print("-" * 40)

    # 4. 後からグラフ化などに使い回せるように、並び替え済みのデータフレームを返す
    return feature_imp_df