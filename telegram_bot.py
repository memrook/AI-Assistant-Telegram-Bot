import asyncio
import logging
import os
from pathlib import Path
import aiofiles

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from yandex_cloud_ml_sdk import YCloudML

from document_processor import DocumentProcessor
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
YANDEX_MODEL_NAME = os.getenv("YANDEX_MODEL_NAME", "yandexgpt-lite")
YANDEX_MODEL_VERSION = os.getenv("YANDEX_MODEL_VERSION", "rc")
DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", "Вы - полезный ассистент.")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.5"))
SHORT_MESSAGE_THRESHOLD = int(os.getenv("SHORT_MESSAGE_THRESHOLD", "300"))

# Глобальные переменные для SDK, индекса и менеджера сессий
sdk = None
session_manager = None
initialization_message = None
document_processor = None


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    if not update.message:
        return
        
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


async def upload_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /upload - инструкция по загрузке документов"""
    if not update.message:
        return
        
    await update.message.reply_text(
        "📤 Загрузка документов\n\n"
        "Отправьте мне документ, и я сохраню его в базу знаний.\n\n"
        "Поддерживаемые форматы:\n"
        "• Markdown (.md) (рекомендуется)\n"
        "• Word документы (.docx)\n"
        "• Текстовые файлы (.txt)\n\n"
        "После загрузки документа я предложу обновить поисковый индекс для включения нового документа в поиск.\n\n"
        "Просто отправьте файл как вложение в следующем сообщении."
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик загруженных документов"""
    if not update.message or not update.message.document:
        return
        
    document = update.message.document
    
    # Проверяем размер файла (максимум 20 МБ)
    max_size = 20 * 1024 * 1024  # 20 МБ
    if document.file_size and document.file_size > max_size:
        await update.message.reply_text(
            "❌ Файл слишком большой. Максимальный размер: 20 МБ"
        )
        return
    
    # Проверяем расширение файла
    file_name = document.file_name or "unknown"
    file_ext = Path(file_name).suffix.lower()
    
    supported_extensions = {'.md', '.docx', '.txt'}
    if file_ext not in supported_extensions:
        await update.message.reply_text(
            f"❌ Неподдерживаемый формат файла: {file_ext}\n\n"
            "Поддерживаемые форматы:\n"
            "• Markdown (.md)\n"
            "• Word документы (.docx)\n"
            "• Текстовые файлы (.txt)"
        )
        return
    
    # Отправляем сообщение о начале загрузки
    status_message = await update.message.reply_text(f"📥 Загружаю файл {file_name}...")
    
    try:
        # Получаем файл от Telegram
        file = await document.get_file()
        
        # Создаем путь для сохранения
        save_dir = Path("./data/md")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Генерируем уникальное имя файла, если файл с таким именем уже существует
        save_path = save_dir / file_name
        counter = 1
        while save_path.exists():
            name_without_ext = Path(file_name).stem
            ext = Path(file_name).suffix
            save_path = save_dir / f"{name_without_ext}_{counter}{ext}"
            counter += 1
        
        # Конвертируем .txt файлы в .md
        if file_ext == '.txt':
            save_path = save_path.with_suffix('.md')
            
        await status_message.edit_text(f"💾 Сохраняю файл {save_path.name}...")
        
        # Скачиваем и сохраняем файл
        await file.download_to_drive(save_path)
        
        # Если это .txt файл, добавляем заголовок в формате markdown
        if file_ext == '.txt':
            # Читаем содержимое
            async with aiofiles.open(save_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Добавляем заголовок и сохраняем
            title = Path(file_name).stem
            markdown_content = f"# {title}\n\n{content}"
            
            async with aiofiles.open(save_path, 'w', encoding='utf-8') as f:
                await f.write(markdown_content)
        
        # Создаем кнопку для обновления индекса
        keyboard = [[InlineKeyboardButton("🔄 Обновить поисковый индекс", callback_data="reindex_after_upload")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await status_message.edit_text(
            f"✅ Файл {save_path.name} успешно сохранен!\n\n"
            f"📁 Путь: {save_path}\n"
            f"📊 Размер: {document.file_size / 1024:.1f} КБ\n\n"
            "Для включения документа в поиск рекомендуется обновить поисковый индекс.",
            reply_markup=reply_markup
        )
        
        logger.info(f"Документ загружен: {save_path}")
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке документа: {e}")
        await status_message.edit_text(
            f"❌ Ошибка при загрузке файла: {str(e)[:100]}...\n\n"
            "Попробуйте загрузить файл еще раз."
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help"""
    if not update.message:
        return
        
    await update.message.reply_text(
        "Вы можете задать мне любой вопрос, и я постараюсь найти ответ в документах.\n\n"
        "Доступные команды:\n"
        "/start - Начать общение с ботом\n"
        "/help - Показать справку\n"
        "/upload - Загрузить новый документ в базу знаний\n"
        "/reset - Сбросить историю разговора\n"
        "/status - Получить статус загрузки документов\n"
        "/cancel - Отменить текущую загрузку документов\n"
        "/history - Показать историю разговора\n"
        "/reindex - Принудительно пересоздать поисковый индекс"
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /reset"""
    if not update.message or not update.effective_user:
        return
        
    user_id = update.effective_user.id
    if session_manager:
        await session_manager.reset_user_thread(user_id)
    await update.message.reply_text("История разговора сброшена. Можете начать новый диалог.")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /status - показывает текущее состояние обработки"""
    if not update.message:
        return
        
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
    """Обработчик команды /cancel - отменяет текущую загрузку документов"""
    if not update.message:
        return
        
    global document_processor

    if not 'document_processor' in globals() or document_processor is None:
        await update.message.reply_text("⚠️ Нет активного процесса загрузки документов.")
        return

    if hasattr(document_processor, 'is_processing') and document_processor.is_processing:
        # Устанавливаем флаг отмены
        document_processor.is_processing = False
        await update.message.reply_text(
            "🛑 Загрузка документов прервана. Загрузка текущего документа будет завершена.")
    else:
        await update.message.reply_text("⚠️ Нет активного процесса загрузки документов.")


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /history - показывает историю разговора"""
    if not update.message or not update.effective_user:
        return
        
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


async def reindex_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /reindex - принудительно пересоздает поисковый индекс"""
    if not update.message:
        return
        
    global document_processor, session_manager, initialization_message
    
    if not document_processor:
        await update.message.reply_text("⚠️ Бот еще не инициализирован. Используйте /start для начала инициализации.")
        return
    
    # Проверяем, что процессор не занят
    if hasattr(document_processor, 'is_processing') and document_processor.is_processing:
        await update.message.reply_text(
            "⚠️ В данный момент уже выполняется загрузка документов. Дождитесь окончания или используйте /cancel для отмены."
        )
        return
    
    # Отправляем сообщение о начале процесса
    progress_message = await update.message.reply_text("🔄 Начинаю пересоздание поискового индекса...")
    initialization_message = progress_message
    
    # Создаем функцию обратного вызова для отображения прогресса
    async def update_progress(message):
        nonlocal progress_message
        if progress_message:
            try:
                await progress_message.edit_text(message)
            except Exception as e:
                # Если сообщение слишком длинное, отправляем новое
                if "message is too long" in str(e).lower() or "message is not modified" in str(e).lower():
                    progress_message = await progress_message.reply_text(message)
    
    # Устанавливаем callback для обновления сообщений
    document_processor.update_callback = update_progress
    
    try:
        # Сначала загружаем файлы, если они еще не загружены
        if not document_processor.files:
            await progress_message.edit_text("🔄 Загружаю документы (Markdown и DOCX файлы) для создания индекса...")
            files = await document_processor.upload_documents()
            if not files:
                await progress_message.edit_text(
                    "⚠️ Не удалось загрузить файлы для индексации. Проверьте наличие документов (.md или .docx файлов) в директории data/md."
                )
                return
        
        # Пересоздаем индекс принудительно
        search_index = await document_processor.create_search_index(force_recreate=True)
        
        if not search_index:
            await progress_message.edit_text(
                "❌ Не удалось создать поисковый индекс. Проверьте логи для получения дополнительной информации."
            )
            return
            
        # Создаем инструмент поиска и обновляем ассистента
        await progress_message.edit_text("🔧 Создаем инструмент поиска и обновляем ассистента...")
        
        # Создаем инструмент поиска
        if sdk:
            search_tool = sdk.tools.search_index(search_index)
        else:
            await progress_message.edit_text("❌ SDK не инициализирован")
            return
        
        # Проверяем текущего ассистента
        if session_manager and session_manager.assistant:
            # Инструкция для ассистента
            instruction = f"""
Инструкция:
1.  Всегда проверяй наличие ссылок на изображения вида ![](_page_X_Picture_Y.jpeg) в найденных текстовых частях контекста.
2.  Если для вопроса пользователя найдено релевантное изображение, упоминай о его наличии и рекомендуй загрузку по предоставленной ссылке (если она
есть).
3.  Отвечай на русском языке, используя в первую очередь информацию из локального контента без внешних источников.

Формат ответа с изображениями:
- Если в текстовых частях контекста есть ссылки на изображения, обязательно добавь в ответ ссылки на них.
- Если есть ссылки, то указывай явно: См. также изображение [./data/book1/images/page_X_Picture_Y.jpeg]"""
            
            try:
                if sdk:
                    # Создаем нового ассистента с новым инструментом поиска
                    model = sdk.models.completions(YANDEX_MODEL_NAME, model_version=YANDEX_MODEL_VERSION)
                    model = model.configure(temperature=DEFAULT_TEMPERATURE)
                    assistant = sdk.assistants.create(
                        model,
                        instruction=instruction,
                        max_prompt_tokens=5000,
                        ttl_days=6,
                        tools=[search_tool]
                    )
                else:
                    await progress_message.edit_text("❌ SDK не инициализирован")
                    return
            except Exception as e:
                logger.error(f"Ошибка при создании ассистента: {e}")
                # Пробуем базовый вариант без инструкции
                if sdk:
                    model = sdk.models.completions(YANDEX_MODEL_NAME, model_version=YANDEX_MODEL_VERSION)
                    model = model.configure(temperature=DEFAULT_TEMPERATURE)
                    assistant = sdk.assistants.create(
                        model,
                        tools=[search_tool]
                    )
                else:
                    await progress_message.edit_text("❌ SDK не инициализирован")
                    return
            
            # Обновляем ассистента в менеджере сессий
            session_manager.assistant = assistant
            
            await progress_message.edit_text("✅ Поисковый индекс успешно пересоздан и ассистент обновлен!")
        else:
            await progress_message.edit_text(
                "⚠️ Поисковый индекс создан, но не удалось обновить ассистента. Попробуйте перезапустить бота командой /start"
            )
    except Exception as e:
        logger.error(f"Ошибка при пересоздании индекса: {e}")
        await progress_message.edit_text(f"❌ Произошла ошибка при пересоздании индекса: {str(e)[:100]}...")


async def run_reindex_process(query):
    """Выполняет пересоздание поискового индекса (для callback запросов)"""
    global document_processor, session_manager
    
    if not document_processor:
        await query.edit_message_text("⚠️ Бот еще не инициализирован.")
        return
    
    # Проверяем, что процессор не занят
    if hasattr(document_processor, 'is_processing') and document_processor.is_processing:
        await query.edit_message_text(
            "⚠️ В данный момент уже выполняется загрузка документов. Дождитесь окончания или используйте /cancel для отмены."
        )
        return
    
    # Отправляем сообщение о начале процесса
    await query.edit_message_text("🔄 Начинаю обновление поискового индекса...")
    
    # Создаем функцию обратного вызова для отображения прогресса
    async def update_progress(message):
        try:
            await query.edit_message_text(message)
        except Exception as e:
            # Если сообщение слишком длинное, отправляем новое
            if "message is too long" in str(e).lower() or "message is not modified" in str(e).lower():
                pass  # Игнорируем ошибки обновления для callback
    
    # Устанавливаем callback для обновления сообщений
    document_processor.update_callback = update_progress
    
    try:
        # Сначала загружаем файлы, если они еще не загружены
        if not document_processor.files:
            await query.edit_message_text("🔄 Загружаю документы для создания индекса...")
            files = await document_processor.upload_documents()
            if not files:
                await query.edit_message_text(
                    "⚠️ Не удалось загрузить файлы для индексации. Проверьте наличие документов в директории data/md."
                )
                return
        
        # Пересоздаем индекс принудительно
        search_index = await document_processor.create_search_index(force_recreate=True)
        
        if not search_index:
            await query.edit_message_text(
                "❌ Не удалось создать поисковый индекс. Проверьте логи для получения дополнительной информации."
            )
            return
            
        # Создаем инструмент поиска и обновляем ассистента
        await query.edit_message_text("🔧 Обновляю ассистента с новым индексом...")
        
        # Создаем инструмент поиска
        if sdk:
            search_tool = sdk.tools.search_index(search_index)
        else:
            await query.edit_message_text("❌ SDK не инициализирован")
            return
        
        # Проверяем текущего ассистента
        if session_manager and session_manager.assistant:
            # Инструкция для ассистента
            instruction = f"""
Инструкция:
1.  Всегда проверяй наличие ссылок на изображения вида ![](_page_X_Picture_Y.jpeg) в найденных текстовых частях контекста.
2.  Если для вопроса пользователя найдено релевантное изображение, упоминай о его наличии и рекомендуй загрузку по предоставленной ссылке (если она
есть).
3.  Отвечай на русском языке, используя в первую очередь информацию из локального контента без внешних источников.

Формат ответа с изображениями:
- Если в текстовых частях контекста есть ссылки на изображения, обязательно добавь в ответ ссылки на них.
- Если есть ссылки, то указывай явно: См. также изображение [./data/book1/images/page_X_Picture_Y.jpeg]"""
            
            try:
                if sdk:
                    # Создаем нового ассистента с новым инструментом поиска
                    model = sdk.models.completions(YANDEX_MODEL_NAME, model_version=YANDEX_MODEL_VERSION)
                    model = model.configure(temperature=DEFAULT_TEMPERATURE)
                    assistant = sdk.assistants.create(
                        model,
                        instruction=instruction,
                        max_prompt_tokens=5000,
                        ttl_days=6,
                        tools=[search_tool]
                    )
                else:
                    await query.edit_message_text("❌ SDK не инициализирован")
                    return
            except Exception as e:
                logger.error(f"Ошибка при создании ассистента: {e}")
                # Пробуем базовый вариант без инструкции
                if sdk:
                    model = sdk.models.completions(YANDEX_MODEL_NAME, model_version=YANDEX_MODEL_VERSION)
                    model = model.configure(temperature=DEFAULT_TEMPERATURE)
                    assistant = sdk.assistants.create(
                        model,
                        tools=[search_tool]
                    )
                else:
                    await query.edit_message_text("❌ SDK не инициализирован")
                    return
            
            # Обновляем ассистента в менеджере сессий
            session_manager.assistant = assistant
            
            await query.edit_message_text("✅ Поисковый индекс успешно обновлен! Новый документ включен в поиск.")
        else:
            await query.edit_message_text(
                "⚠️ Поисковый индекс создан, но не удалось обновить ассистента. Попробуйте перезапустить бота командой /start"
            )
    except Exception as e:
        logger.error(f"Ошибка при обновлении индекса: {e}")
        await query.edit_message_text(f"❌ Произошла ошибка при обновлении индекса: {str(e)[:100]}...")


async def handle_detailed_answer_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик callback для кнопок 'Развернутый ответ' и 'Обновить индекс'"""
    query = update.callback_query
    if not query or not update.effective_user:
        return
        
    # Подтверждаем получение callback
    await query.answer()
    
    user_id = update.effective_user.id
    callback_data = query.data
    
    # Если бот еще не инициализирован, запускаем инициализацию
    global session_manager
    if not session_manager:
        await query.edit_message_text(
            "Начинаю инициализацию бота. Это может занять некоторое время..."
        )
        success = await initialize_yandex_cloud(None)
        if not success:
            await query.edit_message_text(
                "❌ Не удалось инициализировать бота. Пожалуйста, обратитесь к администратору."
            )
            return
    
    # Обрабатываем разные типы callback
    if callback_data == "detailed_answer":
        # Отправляем сообщение о том, что бот обрабатывает запрос
        await query.edit_message_text("Готовлю более развернутый ответ...")
        
        try:
            if session_manager:
                # Отправляем запрос на развернутый ответ
                response = await session_manager.send_message(user_id, "Дай более развернутый ответ")
                
                await query.edit_message_text(response)
            else:
                await query.edit_message_text("Ошибка: сессия не инициализирована")
        except Exception as e:
            logger.error(f"Ошибка при обработке callback для подробного ответа: {e}")
            await query.edit_message_text(
                "Произошла ошибка при получении подробного ответа. Пожалуйста, попробуйте еще раз."
            )
    
    elif callback_data == "reindex_after_upload":
        # Запускаем пересоздание индекса после загрузки документа
        await run_reindex_process(query)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений"""
    if not update.message or not update.effective_user:
        return
        
    # Если бот еще не инициализирован, запускаем инициализацию
    global session_manager, document_processor
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

    if not user_message:
        await update.message.reply_text("Пожалуйста, отправьте текстовое сообщение.")
        return

    # Отправляем сообщение о том, что бот обрабатывает запрос
    processing_message = await update.message.reply_text("Обрабатываю ваш запрос...")

    try:
        # Обычный режим работы с документами
        if session_manager:
            response = await session_manager.send_message(user_id, user_message)
            
            # Проверяем длину ответа и добавляем кнопку, если ответ короткий
            if len(response) < SHORT_MESSAGE_THRESHOLD:
                keyboard = [[InlineKeyboardButton("Развернутый ответ", callback_data="detailed_answer")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await processing_message.edit_text(response, reply_markup=reply_markup)
            else:
                await processing_message.edit_text(response)
        else:
            await processing_message.edit_text("Ошибка: сессия не инициализирована")
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}")
        await processing_message.edit_text(
            "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз."
        )


async def initialize_yandex_cloud(update: Update | None = None):
    """Инициализация Yandex Cloud SDK и создание ассистента"""
    global sdk, session_manager, initialization_message, document_processor
    
    progress_message = None
    if update and update.message:
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
    
    # Инициализация процессора документов
    logger.info("Инициализация процессора документов")
    document_processor = DocumentProcessor(sdk, update_callback=update_progress)
    
    # Проверяем существование индекса
    search_index = None
    existing_index = await document_processor.check_existing_index()
    
    if existing_index:
        # Если существующий индекс найден, используем его
        search_index = existing_index
        if progress_message:
            await progress_message.edit_text(f"✅ Использую существующий поисковый индекс: {search_index.id}")
    else:
        # Если индекс не найден, загружаем документы и создаем новый
        if progress_message:
            await progress_message.edit_text("Существующий индекс не найден. Начинаю загрузку документов (Markdown и DOCX файлов)...")
            
        # Загрузка документов
        files = await document_processor.upload_documents()
        
        # Создание индекса
        if files:
            search_index = await document_processor.create_search_index()
            if not search_index and progress_message:
                await progress_message.edit_text(
                    "⚠️ Не удалось создать поисковый индекс. Бот будет работать в режиме обычного ассистента."
                )
        elif progress_message:
            await progress_message.edit_text(
                "⚠️ Нет файлов для индексации. Проверьте наличие Markdown-файлов в директории data/md. Бот будет работать в режиме обычного ассистента."
            )
    
    # Создание инструмента поиска
    search_tool = None
    if search_index:
        try:
            # Создаем инструмент поиска на основе индекса
            if progress_message:
                await progress_message.edit_text("🔍 Создаю инструмент поиска на основе индекса...")
            
            search_tool = sdk.tools.search_index(search_index)
            logger.info(f"Инструмент поиска создан на основе индекса: {search_index.id}")
        except Exception as e:
            logger.error(f"Ошибка при создании инструмента поиска: {e}")
            if progress_message:
                await progress_message.edit_text(
                    f"⚠️ Ошибка при создании инструмента поиска: {str(e)[:100]}...\n"
                    "Бот будет работать в режиме обычного ассистента."
                )
    
    # Создание ассистента (с инструментом поиска или без него)
    if progress_message:
        await progress_message.edit_text("🔧 Создаем ассистента...")
    
    logger.info("Создаем ассистента")

    # Добавляем инструмент поиска, только если он был создан
    tools = [search_tool] if search_tool else []
    mode = "с гибридным поиском по документам" if search_tool else "в режиме обычного ассистента (без поиска по документам)"
    logger.info(f"Создаем ассистента {mode}")

    # Инструкция для ассистента с поиском по документам
    instruction = f"""
Инструкция:
1.  Всегда проверяй наличие ссылок на изображения вида ![](_page_X_Picture_Y.jpeg) в найденных текстовых частях контекста.
2.  Если для вопроса пользователя найдено релевантное изображение, упоминай о его наличии и рекомендуй загрузку по предоставленной ссылке (если она
есть).
3.  Отвечай на русском языке, используя в первую очередь информацию из локального контента без внешних источников.

Формат ответа с изображениями:
- Если в текстовых частях контекста есть ссылки на изображения, обязательно добавь в ответ ссылки на них.
- Если есть ссылки, то указывай явно: См. также изображение [./data/book1/images/page_X_Picture_Y.jpeg]."""

    try:
        # Создание ассистента с правильным API
        if search_tool:
            model = sdk.models.completions(YANDEX_MODEL_NAME, model_version=YANDEX_MODEL_VERSION)
            model = model.configure(temperature=DEFAULT_TEMPERATURE)
            assistant = sdk.assistants.create(
                model,
                instruction=instruction,
                temperature=0.5,
                max_prompt_tokens=5000,
                ttl_days=6,
                expiration_policy="static",
                tools=tools
            )
            logger.info("Ассистент создан с инструкцией и инструментом поиска")
        # else:
        #     # Создание обычного ассистента без инструментов поиска
        #     assistant = sdk.assistants.create(
        #         "yandexgpt-lite", 
        #         model_version="rc",
        #         temperature=DEFAULT_TEMPERATURE,
        #         instruction=instruction
        #     )
        #     logger.info("Создан обычный ассистент без инструментов поиска")
    except Exception as e:
        logger.error(f"Ошибка при создании ассистента: {e}")
        # Пробуем создать базового ассистента в случае ошибки
        try:
            assistant = sdk.assistants.create("yandexgpt", temperature=DEFAULT_TEMPERATURE)
            logger.warning("Создан базовый ассистент из-за ошибки")
        except Exception as e2:
            logger.error(f"Критическая ошибка при создании ассистента: {e2}")
            return False

    # Инициализация менеджера сессий
    if progress_message:
        await progress_message.edit_text(f"🔧 Инициализация менеджера сессий... Ассистент работает {mode}.")

    logger.info("Инициализация менеджера сессий")
    session_manager = SessionManager(sdk, assistant)

    if progress_message:
        await progress_message.edit_text(f"✅ Инициализация завершена! Бот готов к работе {mode}.")

    return True


async def main():
    """Запуск бота"""
    # Создание бота
    logger.info("Запуск бота")
    
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN не установлен")
        return None
        
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Регистрация обработчиков команд
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("cancel", cancel_command))
    application.add_handler(CommandHandler("history", history_command))
    application.add_handler(CommandHandler("reindex", reindex_command))
    application.add_handler(CommandHandler("upload", upload_command)) # Регистрируем обработчик команды /upload

    # Регистрация обработчика текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document)) # Регистрируем обработчик загруженных документов

    # Регистрация обработчика callback-запросов
    application.add_handler(CallbackQueryHandler(handle_detailed_answer_callback))

    # Настраиваем graceful shutdown
    application.post_shutdown = shutdown_handler

    # Запуск бота (без блокировки)
    try:
        await application.initialize()
        await application.start()
        if application.updater:
            await application.updater.start_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

            # Сообщаем, что бот запущен и готов
            logger.info("Бот успешно запущен и готов к работе")

            # Возвращаем объект приложения, чтобы иметь возможность корректно завершить его позже
            return application
        else:
            logger.error("Updater не инициализирован")
            return None
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
