import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def test_model_training():
    # Загрузка данных
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Проверка предсказаний
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
    assert all(pred in [0, 1, 2] for pred in predictions)  # Проверка, что предсказания в правильном диапазоне 