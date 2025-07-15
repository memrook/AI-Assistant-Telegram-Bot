#!/bin/bash

# Скрипт для быстрого развертывания AI-Assistant-MD в Docker

set -e  # Прерывать выполнение при ошибках

echo "🚀 Развертывание AI-Assistant-MD в Docker..."

# Проверяем наличие Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен. Установите Docker и повторите попытку."
    exit 1
fi

# Проверяем наличие Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose не установлен. Установите Docker Compose и повторите попытку."
    exit 1
fi

# Проверяем наличие .env файла
if [ ! -f .env ]; then
    echo "⚠️  Файл .env не найден. Создаем из примера..."
    if [ -f env.example ]; then
        cp env.example .env
        echo "📝 Отредактируйте файл .env и заполните необходимые переменные:"
        echo "   - YANDEX_API_KEY"
        echo "   - YANDEX_FOLDER_ID" 
        echo "   - TELEGRAM_BOT_TOKEN"
        echo ""
        echo "После заполнения .env файла запустите скрипт повторно."
        exit 1
    else
        echo "❌ Файл env.example не найден. Создайте .env файл вручную."
        exit 1
    fi
fi

# Проверяем наличие директории с документами
if [ ! -d "data/md" ]; then
    echo "📁 Создаем директорию data/md для документов..."
    mkdir -p data/md
    echo "⚠️  Поместите ваши документы (.md или .docx файлы) в директорию data/md/"
fi

# Проверяем наличие документов
if [ -z "$(ls -A data/md 2>/dev/null)" ]; then
    echo "⚠️  Директория data/md пуста. Поместите туда документы перед запуском."
    echo "Поддерживаемые форматы: .md, .docx"
fi

echo "🔨 Сборка Docker образа..."
docker-compose build

echo "🏃 Запуск контейнера..."
docker-compose up -d

echo "✅ Развертывание завершено!"
echo ""
echo "📊 Управление контейнером:"
echo "  Просмотр логов:     docker-compose logs -f"
echo "  Остановка:          docker-compose down"
echo "  Перезапуск:         docker-compose restart"
echo "  Статус:             docker-compose ps"
echo ""
echo "🤖 Ваш Telegram бот готов к работе!" 