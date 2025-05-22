import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

def train_model():
    # Загрузка данных
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Настройка MLFlow
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("iris_classification")

    with mlflow.start_run():
        # Параметры модели
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        }
        
        # Логирование параметров
        mlflow.log_params(params)
        
        # Обучение модели
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Предсказания и метрики
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Логирование метрик
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Сохранение модели
        mlflow.sklearn.log_model(model, "iris_model")
        
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 Score: {f1:.3f}")

if __name__ == "__main__":
    train_model() 