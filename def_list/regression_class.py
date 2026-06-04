from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ModelAuditor:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        # データをクラスの中に固定する（読み込み直しの手間がなくなる）
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.models = {}
        self.results = {}
        self.X_columns = X.columns

    def run(self, dt_max_depth=3):
        # 実行時にモデルを学習し、結果を自分の中に保存する
        pipelines = {
            'LogisticRegression': Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())]),
            'DecisionTree': Pipeline([('scaler', StandardScaler()), ('dt', DecisionTreeClassifier(max_depth=dt_max_depth))]),
            'RandomForest': Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier())])
        }
        
        for name, pipe in pipelines.items():
            pipe.fit(self.X_train, self.y_train)
            score = accuracy_score(self.y_val, pipe.predict(self.X_val))
            self.results[name] = score
            self.models[name] = pipe
            print(f"{name} の正解率: {score:.4f}")

    def plot_importances(self):
        # 可視化ロジックも自分の中に持つ
        for name, pipe in self.models.items():
            model = list(pipe.named_steps.values())[1]
            importances = abs(model.coef_[0]) if name == 'LogisticRegression' else model.feature_importances_
            
            df_imp = pd.DataFrame({'Feature': self.X_columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
            
            plt.figure(figsize=(8, 4))
            sns.barplot(x='Importance', y='Feature', data=df_imp.head(10), palette='magma')
            plt.title(f'Top 10 Features: {name}')
            plt.show()