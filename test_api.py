#!/usr/bin/env python3
"""
Скрипт для тестирования работоспособности Yandex Cloud API
Помогает диагностировать проблемы с подключением и конфигурацией
"""

import os
import logging
import asyncio
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def test_yandex_cloud_connection():
    """Тестирует подключение к Yandex Cloud"""
    
    print("🔍 Тестирование API Yandex Cloud...")
    
    # Загрузка переменных окружения
    load_dotenv()
    
    YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
    YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
    
    # Проверка переменных окружения
    print("\n📋 Проверка переменных окружения:")
    if not YANDEX_API_KEY:
        print("❌ YANDEX_API_KEY не установлен")
        return False
    else:
        masked_key = YANDEX_API_KEY[:10] + "..." + YANDEX_API_KEY[-4:] if len(YANDEX_API_KEY) > 14 else "***"
        print(f"✅ YANDEX_API_KEY: {masked_key}")
    
    if not YANDEX_FOLDER_ID:
        print("❌ YANDEX_FOLDER_ID не установлен")
        return False
    else:
        print(f"✅ YANDEX_FOLDER_ID: {YANDEX_FOLDER_ID}")
    
    # Инициализация SDK
    print("\n🔧 Инициализация SDK...")
    try:
        sdk = YCloudML(
            folder_id=YANDEX_FOLDER_ID,
            auth=YANDEX_API_KEY,
        )
        print("✅ SDK инициализирован успешно")
    except Exception as e:
        print(f"❌ Ошибка инициализации SDK: {e}")
        return False
    
    # Тест создания треда
    print("\n🧵 Тестирование создания треда...")
    try:
        thread = sdk.threads.create()
        print(f"✅ Тред создан успешно: {thread.id}")
    except Exception as e:
        print(f"❌ Ошибка создания треда: {e}")
        return False
    
    # Тест создания ассистента
    print("\n🤖 Тестирование создания ассистента...")
    try:
        model = sdk.models.completions("yandexgpt-lite", model_version="rc")
        model = model.configure(temperature=0.5)
        assistant = sdk.assistants.create(
            model,
            instruction="Ты тестовый ассистент. Отвечай кратко.",
            max_prompt_tokens=1000,
            ttl_days=1,
            expiration_policy="static"
        )
        print(f"✅ Ассистент создан успешно: {assistant.id}")
    except Exception as e:
        print(f"❌ Ошибка создания ассистента: {e}")
        return False
    
    # Тест отправки сообщения
    print("\n💬 Тестирование отправки сообщения...")
    try:
        thread.write("Привет! Это тестовое сообщение.")
        print("✅ Сообщение записано в тред")
        
        # Запуск ассистента
        print("🚀 Запуск ассистента...")
        run = assistant.run(thread)
        
        # Ожидание результата с таймаутом
        print("⏳ Ожидание ответа...")
        result = run.wait()
        
        # Проверка результата
        if hasattr(result, 'message') and result.message:
            response = str(result.message)
            print(f"✅ Получен ответ: {response[:100]}{'...' if len(response) > 100 else ''}")
        elif result:
            response = str(result)
            print(f"✅ Получен ответ (альтернативный формат): {response[:100]}{'...' if len(response) > 100 else ''}")
        else:
            print("⚠️ Ответ получен, но он пустой")
            
    except Exception as e:
        print(f"❌ Ошибка при тестировании сообщений: {e}")
        return False
    
    print("\n🎉 Все тесты пройдены успешно! API работает корректно.")
    return True

async def test_document_processing():
    """Тестирует обработку документов"""
    
    print("\n📄 Тестирование обработки документов...")
    
    # Проверка наличия директории с документами
    doc_dir = "./data/md"
    if not os.path.exists(doc_dir):
        print(f"⚠️ Директория {doc_dir} не существует")
        return False
    
    # Подсчет файлов
    md_files = len([f for f in os.listdir(doc_dir) if f.endswith('.md')])
    docx_files = len([f for f in os.listdir(doc_dir) if f.endswith('.docx')])
    
    print(f"📊 Найдено документов:")
    print(f"   • Markdown (.md): {md_files}")
    print(f"   • DOCX (.docx): {docx_files}")
    print(f"   • Всего: {md_files + docx_files}")
    
    if md_files + docx_files == 0:
        print("⚠️ Нет документов для обработки")
        return False
    
    print("✅ Документы для обработки найдены")
    return True

async def main():
    """Основная функция тестирования"""
    
    print("=" * 60)
    print("🧪 ТЕСТИРОВАНИЕ AI-ASSISTANT-MD")
    print("=" * 60)
    
    # Тест API
    api_ok = await test_yandex_cloud_connection()
    
    # Тест документов
    docs_ok = await test_document_processing()
    
    print("\n" + "=" * 60)
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    print(f"🔗 API подключение: {'✅ OK' if api_ok else '❌ FAIL'}")
    print(f"📄 Документы: {'✅ OK' if docs_ok else '❌ FAIL'}")
    
    if api_ok and docs_ok:
        print("\n🎉 Все системы готовы к работе!")
        print("💡 Можете запускать основное приложение:")
        print("   • Обычный запуск: python main.py")
        print("   • Docker запуск: ./deploy.sh")
    else:
        print("\n❌ Обнаружены проблемы:")
        if not api_ok:
            print("   • Проверьте настройки API в .env файле")
            print("   • Убедитесь в корректности YANDEX_API_KEY и YANDEX_FOLDER_ID")
        if not docs_ok:
            print("   • Добавьте документы (.md или .docx) в директорию data/md/")
    
    return api_ok and docs_ok

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 