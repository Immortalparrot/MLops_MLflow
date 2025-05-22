# Используем официальный Python-образ
FROM python:3.11

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY . .

# Устанавливаем зависимости
RUN pip install --upgrade pip && pip install -r requirements.txt

# Открываем порт для MLflow
EXPOSE 5001

# По умолчанию запускаем MLflow сервер
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5001"] 