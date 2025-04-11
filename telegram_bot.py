import os
import logging
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.search_indexes import (
    HybridSearchIndexType,
    StaticIndexChunkingStrategy,
    ReciprocalRankFusionIndexCombinationStrategy,
)

from pdf_converter import PDFProcessor
from session_manager import SessionManager

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", "Вы - полезный ассистент.")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.5"))

# Глобальные переменные для SDK, индекса и менеджера сессий
sdk = None
session_manager = None
initialization_message = None
pdf_processor = None

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    # Если бот еще не инициализирован, запускаем инициализацию
    global session_manager
    if not session_manager:
        await update.message.reply_text(
            "Начинаю инициализацию бота. Это может занять некоторое время..."
        )
        success = await initialize_yandex_cloud(update)
        if not success:
            await update.message.reply_text(
                "❌ Не удалось инициализировать бота. Пожалуйста, обратитесь к администратору."
            )
            return
    
    await update.message.reply_text(
        "Привет! Я AI-ассистент, который поможет вам найти информацию в документах. "
        "Задайте мне вопрос, и я постараюсь на него ответить."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help"""
    await update.message.reply_text(
        "Вы можете задать мне любой вопрос, и я постараюсь найти ответ в документах.\n\n"
        "Доступные команды:\n"
        "/start - Начать общение с ботом\n"
        "/help - Показать справку\n"
        "/reset - Сбросить историю разговора\n"
        "/status - Получить статус обработки документов\n"
        "/cancel - Отменить текущую обработку документов\n"
        "/history - Показать историю разговора"
    )

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /reset"""
    user_id = update.effective_user.id
    await session_manager.reset_user_thread(user_id)
    await update.message.reply_text("История разговора сброшена. Можете начать новый диалог.")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /status - показывает текущее состояние обработки"""
    global initialization_message, session_manager
    
    if not session_manager:
        await update.message.reply_text("⚠️ Бот еще не инициализирован. Используйте /start для начала инициализации.")
        return
        
    if initialization_message:
        current_text = initialization_message.text
        await update.message.reply_text(f"📊 Текущий статус:\n\n{current_text}")
    else:
        await update.message.reply_text("✅ Инициализация завершена. Бот готов к работе.")

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /cancel - отменяет текущую обработку документов"""
    global pdf_processor
    
    if not 'pdf_processor' in globals() or pdf_processor is None:
        await update.message.reply_text("⚠️ Нет активного процесса обработки документов.")
        return
    
    if hasattr(pdf_processor, 'is_processing') and pdf_processor.is_processing:
        # Устанавливаем флаг отмены
        pdf_processor.is_processing = False
        await update.message.reply_text("🛑 Обработка документов прервана. Обработка текущего документа будет завершена.")
    else:
        await update.message.reply_text("⚠️ Нет активного процесса обработки документов.")

async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /history - показывает историю разговора"""
    global session_manager
    
    if not session_manager:
        await update.message.reply_text("⚠️ Бот еще не инициализирован. Используйте /start для начала инициализации.")
        return
    
    user_id = update.effective_user.id
    history = session_manager.get_conversation_history(user_id)
    
    if history == "История разговора пуста.":
        await update.message.reply_text("У вас еще нет истории разговора с ботом.")
    else:
        # Разбиваем историю на части, если она слишком длинная
        max_length = 4000  # Максимальная длина сообщения в Telegram
        
        if len(history) <= max_length:
            await update.message.reply_text(f"📝 История разговора:\n\n{history}")
        else:
            # Разбиваем на части
            parts = []
            current_part = ""
            
            for line in history.split("\n\n"):
                if len(current_part) + len(line) + 2 > max_length:
                    parts.append(current_part)
                    current_part = line
                else:
                    if current_part:
                        current_part += "\n\n" + line
                    else:
                        current_part = line
            
            if current_part:
                parts.append(current_part)
            
            # Отправляем части
            await update.message.reply_text(f"📝 История разговора (часть 1/{len(parts)}):\n\n{parts[0]}")
            
            for i, part in enumerate(parts[1:], 2):
                await update.message.reply_text(f"📝 История разговора (часть {i}/{len(parts)}):\n\n{part}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений"""
    # Если бот еще не инициализирован, запускаем инициализацию
    global session_manager
    if not session_manager:
        await update.message.reply_text(
            "Начинаю инициализацию бота. Это может занять некоторое время..."
        )
        success = await initialize_yandex_cloud(update)
        if not success:
            await update.message.reply_text(
                "❌ Не удалось инициализировать бота. Пожалуйста, обратитесь к администратору."
            )
            return
    
    user_id = update.effective_user.id
    user_message = update.message.text
    
    # Отправляем сообщение о том, что бот обрабатывает запрос
    processing_message = await update.message.reply_text("Обрабатываю ваш запрос...")
    
    try:
        # Получаем ответ от ассистента
        response = await session_manager.send_message(user_id, user_message)
        
        # Отправляем ответ пользователю
        await processing_message.edit_text(response)
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}")
        await processing_message.edit_text(
            "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз."
        )

async def initialize_yandex_cloud(update: Update = None):
    """Инициализация Yandex Cloud SDK и создание ассистента"""
    global sdk, session_manager, initialization_message, pdf_processor
    
    progress_message = None
    if update:
        progress_message = await update.message.reply_text("Инициализация Yandex Cloud SDK...")
        initialization_message = progress_message
    
    logger.info("Инициализация Yandex Cloud SDK")
    sdk = YCloudML(
        folder_id=YANDEX_FOLDER_ID,
        auth=YANDEX_API_KEY,
    )
    
    # Создаем функцию обратного вызова для отображения прогресса в Telegram
    async def update_progress(message):
        nonlocal progress_message
        if progress_message:
            try:
                await progress_message.edit_text(message)
            except Exception as e:
                # Если сообщение слишком длинное, отправляем новое
                if "message is too long" in str(e).lower() or "message is not modified" in str(e).lower():
                    progress_message = await progress_message.reply_text(message)
    
    # Конвертирование PDF и создание индекса
    logger.info("Начинаем обработку PDF-файлов")
    pdf_processor = PDFProcessor(sdk, update_callback=update_progress)
    
    # Обработка документов
    files = await pdf_processor.convert_and_upload_pdfs()
    
    # Создание инструмента поиска
    search_tool = None
    if files:
        # Пытаемся создать поисковый индекс
        try:
            search_index = await pdf_processor.create_search_index()
            if search_index:
                if progress_message:
                    await progress_message.edit_text("🔍 Поисковый индекс создан успешно. Создаем инструмент поиска...")
                    
                # Создаем инструмент поиска
                search_tool = sdk.tools.search_index(search_index)
                logger.info(f"Поисковый индекс успешно создан, ID: {search_index.id}")
            else:
                if progress_message:
                    await progress_message.edit_text("⚠️ Не удалось создать поисковый индекс. Бот будет работать в режиме обычного ассистента.")
        except Exception as e:
            logger.error(f"Ошибка при создании поискового индекса: {e}")
            if progress_message:
                await progress_message.edit_text(f"⚠️ Ошибка при создании поискового индекса: {str(e)[:100]}...\nБот будет работать в режиме обычного ассистента.")
    else:
        if progress_message:
            await progress_message.edit_text("⚠️ Нет файлов для индексации. Бот будет работать в режиме обычного ассистента.")
    
    # Создание ассистента (с инструментом поиска или без него)
    if progress_message:
        await progress_message.edit_text("🔧 Создаем ассистента...")
    
    logger.info("Создаем ассистента")
    model = sdk.models.completions("yandexgpt", model_version="rc")
    
    # Настраиваем модель с правильными параметрами
    try:
        # В новой версии SDK параметры могут отличаться
        # Пробуем разные варианты настройки в порядке приоритета
        try:
            # Вариант 1: попробуем использовать параметр prompt_template для системного промпта
            model = model.configure(
                temperature=DEFAULT_TEMPERATURE,
                prompt_template={"system_prompt": DEFAULT_SYSTEM_PROMPT}
            )
            logger.info("Модель настроена с использованием prompt_template")
        except (TypeError, ValueError):
            try:
                # Вариант 2: возможно, нужно использовать просто temperature
                model = model.configure(temperature=DEFAULT_TEMPERATURE)
                logger.info(f"Модель настроена только с температурой {DEFAULT_TEMPERATURE}")
                
                # Попробуем добавить системный промпт через другие методы, если они доступны
                if hasattr(model, 'with_system_prompt'):
                    model = model.with_system_prompt(DEFAULT_SYSTEM_PROMPT)
                    logger.info("Добавлен системный промпт через метод with_system_prompt")
            except Exception as e:
                logger.warning(f"Не удалось настроить модель: {e}. Используем модель по умолчанию.")
    except Exception as e:
        logger.error(f"Ошибка при настройке модели: {e}")
        # Продолжаем работу с моделью по умолчанию, без дополнительных настроек
    
    # Добавляем инструмент поиска, только если он был создан
    tools = [search_tool] if search_tool else []
    mode = "с гибридным поиском по документам" if search_tool else "в режиме обычного ассистента (без поиска по документам)"
    logger.info(f"Создаем ассистента {mode}")
    
    # Инструкция для ассистента с поиском по документам
    instruction = """Вы - умный ассистент агронома, который помогает пользователям находить информацию в документах.

При ответе на вопросы следуйте этим правилам:
1. Всегда используйте инструмент поиска для нахождения релевантной информации в документах.
2. Если в документах есть ответ на вопрос, используйте ТОЛЬКО информацию из документов.
3. Если информации в документах недостаточно, укажите это и дополните ответ своими знаниями.
4. Если в документах нет ответа на вопрос, сообщите об этом и ответьте на основе своих знаний.
5. Всегда будьте вежливы, информативны и старайтесь дать максимально полный ответ.
6. Отвечайте на русском языке, даже если вопрос задан на другом языке.
7. Если вопрос двусмысленный или неясный, попросите уточнить детали.
8. При ответе на основе документов, указывайте источник информации.

Ваша основная задача - предоставлять точную и полезную информацию из загруженных документов."""
    
    try:
        # Вариант 1: создание ассистента с указанными инструментами, моделью и инструкцией
        if search_tool:
            try:
                assistant = sdk.assistants.create(model, tools=tools, instruction=instruction)
                logger.info("Ассистент создан с инструкцией и инструментом поиска")
            except TypeError as e:
                logger.warning(f"Не удалось создать ассистента с инструкцией: {e}. Пробуем альтернативный вариант.")
                try:
                    # Проверяем другие возможные параметры для instruction
                    if hasattr(model, 'with_system_prompt'):
                        model = model.with_system_prompt(instruction)
                        assistant = sdk.assistants.create(model, tools=tools)
                        logger.info("Ассистент создан с системным промптом и инструментом поиска")
                    else:
                        assistant = sdk.assistants.create(model, tools=tools)
                        logger.info("Ассистент создан только с инструментом поиска (без инструкции)")
                except Exception as e2:
                    logger.error(f"Ошибка при создании ассистента с инструментом поиска: {e2}")
                    assistant = sdk.assistants.create(model)
                    logger.warning("Ассистент создан без инструментов поиска из-за ошибки")
        else:
            # Создание обычного ассистента без инструментов поиска
            assistant = sdk.assistants.create(model)
            logger.info("Создан обычный ассистент без инструментов поиска")
    except Exception as e:
        logger.error(f"Ошибка при создании ассистента: {e}")
        # Пробуем создать базового ассистента в случае ошибки
        assistant = sdk.assistants.create(model)
        logger.warning("Создан базовый ассистент из-за ошибки")
    
    # Инициализация менеджера сессий
    if progress_message:
        await progress_message.edit_text(f"🔧 Инициализация менеджера сессий... Ассистент работает {mode}.")
    
    logger.info("Инициализация менеджера сессий")
    session_manager = SessionManager(sdk, assistant)
    
    if progress_message:
        await progress_message.edit_text(f"✅ Инициализация завершена! Бот готов к работе {mode}.")
    
    return True

async def main() -> None:
    """Запуск бота"""
    # Создание бота
    logger.info("Запуск бота")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Регистрация обработчиков команд
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("cancel", cancel_command))
    application.add_handler(CommandHandler("history", history_command))
    
    # Регистрация обработчика текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Настраиваем graceful shutdown
    application.post_shutdown = shutdown_handler
    
    # Запуск бота (без блокировки)
    try:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
        
        # Сообщаем, что бот запущен и готов
        logger.info("Бот успешно запущен и готов к работе")
        
        # Возвращаем объект приложения, чтобы иметь возможность корректно завершить его позже
        return application
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}")
        # Пытаемся остановить уже запущенные компоненты
        try:
            if hasattr(application, 'updater') and application.updater:
                await application.updater.stop()
            await application.stop()
            await application.shutdown()
        except Exception as stop_error:
            logger.error(f"Ошибка при остановке бота: {stop_error}")
        raise e

async def shutdown_handler(application: Application) -> None:
    """Обработчик завершения работы бота"""
    logger.info("Выполняем дополнительные действия при остановке бота...")
    # Здесь можно добавить любую логику, которая должна выполняться при завершении
    # Например, закрытие соединений с базой данных или другими сервисами
    logger.info("Дополнительные действия выполнены, бот успешно завершен")

if __name__ == "__main__":
    asyncio.run(main()) 