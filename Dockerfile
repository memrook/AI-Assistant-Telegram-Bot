# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Создаем пользователя для приложения (не root)
RUN useradd --create-home --shell /bin/bash app

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY --chown=app:app . .

# Создаем необходимые директории
RUN mkdir -p data/md data && \
    chown -R app:app /app

# Переключаемся на пользователя app
USER app

# Открываем порт (если понадобится для веб-интерфейса в будущем)
EXPOSE 8000

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Команда по умолчанию
CMD ["python", "main.py"] 