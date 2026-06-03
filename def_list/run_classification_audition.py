from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_classification_audition(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipelines = {
        'LogisticRegression': Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())]),
        'DecisionTree': Pipeline([('scaler', StandardScaler()), ('dt', DecisionTreeClassifier(max_depth=3))]),
        'RandomForest': Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier())])
    }
    
    results = {}
    print("---分類オーディション開始 ---")
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        score = accuracy_score(y_val, pipe.predict(X_val))
        print(f"{name} の正解率: {score:.4f}")
        results[name] = pipe
    return results