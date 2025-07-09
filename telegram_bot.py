import os
import logging
import asyncio
from dotenv import load_dotenv
from telegram import Update, InputFile
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
        "/history - Показать историю разговора\n"
        "/reindex - Принудительно пересоздать поисковый индекс\n"
        "/extract_images <имя_файла> - Извлечь изображения из PDF (например: /extract_images example.pdf)"
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
        await update.message.reply_text(
            "🛑 Обработка документов прервана. Обработка текущего документа будет завершена.")
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


async def reindex_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /reindex - принудительно пересоздает поисковый индекс"""
    global pdf_processor, session_manager, initialization_message
    
    if not pdf_processor:
        await update.message.reply_text("⚠️ Бот еще не инициализирован. Используйте /start для начала инициализации.")
        return
    
    # Проверяем, что процессор не занят
    if hasattr(pdf_processor, 'is_processing') and pdf_processor.is_processing:
        await update.message.reply_text(
            "⚠️ В данный момент уже выполняется обработка документов. Дождитесь окончания или используйте /cancel для отмены."
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
    pdf_processor.update_callback = update_progress
    
    try:
        # Сначала загружаем файлы, если они еще не загружены
        if not pdf_processor.files:
            await progress_message.edit_text("🔄 Загружаю файлы для создания индекса...")
            files = await pdf_processor.convert_and_upload_pdfs()
            if not files:
                await progress_message.edit_text(
                    "⚠️ Не удалось загрузить файлы для индексации. Проверьте наличие PDF-файлов в директории data/docs."
                )
                return
        
        # Пересоздаем индекс принудительно
        search_index = await pdf_processor.create_search_index(force_recreate=True)
        
        if not search_index:
            await progress_message.edit_text(
                "❌ Не удалось создать поисковый индекс. Проверьте логи для получения дополнительной информации."
            )
            return
            
        # Создаем инструмент поиска и обновляем ассистента
        await progress_message.edit_text("🔧 Создаем инструмент поиска и обновляем ассистента...")
        
        # Создаем инструмент поиска
        search_tool = sdk.tools.search_index(search_index)
        
        # Проверяем текущего ассистента
        if session_manager and session_manager.assistant:
            # Создаем нового ассистента с обновленным инструментом поиска
            model = sdk.models.completions("yandexgpt-lite", model_version="rc")
            
            # Настроим модель (используем тот же код, что и при начальной инициализации)
            try:
                # Применяем настройки модели из основного кода
                model = model.configure(temperature=DEFAULT_TEMPERATURE)
            except Exception as e:
                logger.error(f"Ошибка при настройке модели: {e}")
            
            # Создаем нового ассистента с новым инструментом поиска
            instruction = """Вы - умный ассистент агронома, который помогает пользователям находить информацию в документах.

Давай ответ по следующей структуре:
- первый абзац - краткая суть рассматриваемого вопроса
- далее подробный ответ с разбивкой по пунктам
- если есть возможность приводить примеры - приводи
- в конце вывод

При ответе на вопросы следуйте этим правилам:
1. Всегда используйте инструмент поиска для нахождения релевантной информации в документах.
2. Если в документах есть ответ на вопрос, используйте ТОЛЬКО информацию из документов.
3. Если информации в документах недостаточно, укажите это и дополните ответ своими знаниями.
4. Если в документах нет ответа на вопрос, сообщите об этом и ответьте на основе своих знаний.
5. Всегда будьте вежливы, информативны и старайтесь дать максимально полный ответ.
6. Отвечайте на русском языке, даже если вопрос задан на другом языке.
7. Если вопрос двусмысленный или неясный, попросите уточнить детали.
8. При ответе на основе документов, указывайте источник информации.
9. Если в документах ссылка на рисунок, то приложи ссылку на этот рисунок!

Ваша основная задача - предоставлять точную и полезную информацию из загруженных документов."""
            
            try:
                assistant = sdk.assistants.create(model, tools=[search_tool], instruction=instruction)
            except Exception as e:
                logger.error(f"Ошибка при создании ассистента: {e}")
                # Пробуем базовый вариант без инструкции
                assistant = sdk.assistants.create(model, tools=[search_tool])
            
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


async def extract_images_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /extract_images - извлекает изображения из PDF и отправляет их в чат"""
    global pdf_processor
    
    if not pdf_processor:
        await update.message.reply_text("⚠️ Бот еще не инициализирован. Используйте /start для начала инициализации.")
        return
    
    # Проверяем, что процессор не занят
    if hasattr(pdf_processor, 'is_processing') and pdf_processor.is_processing:
        await update.message.reply_text(
            "⚠️ В данный момент уже выполняется обработка документов. Дождитесь окончания или используйте /cancel для отмены."
        )
        return
    
    # Проверяем, что указано имя файла
    if not context.args or len(context.args) < 1:
        await update.message.reply_text(
            "❌ Не указано имя PDF-файла. Пример использования: /extract_images example.pdf"
        )
        return
    
    filename = context.args[0]
    docs_dir = "./data/docs"
    pdf_path = os.path.join(docs_dir, filename)
    
    # Проверяем существование файла
    if not os.path.exists(pdf_path):
        await update.message.reply_text(
            f"❌ Файл {filename} не найден в директории {docs_dir}. "
            f"Убедитесь, что файл существует и попробуйте снова."
        )
        return
    
    # Отправляем сообщение о начале процесса
    progress_message = await update.message.reply_text(
        f"🔄 Начинаю извлечение изображений из файла {filename}..."
    )
    
    try:
        # Извлекаем изображения из PDF
        pdf_processor.is_processing = True
        images_info = await pdf_processor.extract_images_from_pdf(pdf_path)
        pdf_processor.is_processing = False
        
        if not images_info:
            await progress_message.edit_text(
                f"❌ Не удалось извлечь изображения из файла {filename}."
            )
            return
        
        # Сохраняем информацию об изображениях в контексте чата для использования в обработчике обычных сообщений
        context.chat_data["last_extracted_images"] = images_info
        
        # Отправляем информацию о результатах
        result_message = (
            f"✅ Успешно извлечены изображения из файла {filename}:\n"
            f"📄 Страниц: {len(images_info['pages'])}\n"
            f"📊 Таблиц: {len(images_info['tables'])}\n"
            f"🖼 Рисунков: {len(images_info['pictures'])}\n\n"
            f"Отправляю изображения..."
        )
        await progress_message.edit_text(result_message)
        
        # Отправляем изображения в чат (максимум 10, чтобы не спамить)
        MAX_IMAGES = 10
        sent_images = 0
        
        # Отправляем изображения страниц (не более MAX_IMAGES/3)
        max_pages = min(len(images_info['pages']), MAX_IMAGES // 3)
        for i in range(max_pages):
            page_info = images_info['pages'][i]
            with open(page_info['file_path'], 'rb') as photo:
                await update.message.reply_photo(
                    photo=photo,
                    caption=f"📄 Страница {page_info['page_no']} из документа {images_info['document_name']}"
                )
            sent_images += 1
        
        # Отправляем изображения таблиц (не более MAX_IMAGES/3)
        max_tables = min(len(images_info['tables']), MAX_IMAGES // 3)
        for i in range(max_tables):
            table_info = images_info['tables'][i]
            with open(table_info['file_path'], 'rb') as photo:
                await update.message.reply_photo(
                    photo=photo,
                    caption=f"📊 Таблица {table_info['table_no']} из документа {images_info['document_name']}"
                )
            sent_images += 1
        
        # Отправляем изображения рисунков (оставшиеся слоты до MAX_IMAGES)
        remaining_slots = MAX_IMAGES - sent_images
        max_pictures = min(len(images_info['pictures']), remaining_slots)
        for i in range(max_pictures):
            picture_info = images_info['pictures'][i]
            with open(picture_info['file_path'], 'rb') as photo:
                await update.message.reply_photo(
                    photo=photo,
                    caption=f"🖼 Рисунок {picture_info['picture_no']} из документа {images_info['document_name']}"
                )
            sent_images += 1
        
        # Если есть неотправленные изображения, сообщаем об этом
        total_images = len(images_info['pages']) + len(images_info['tables']) + len(images_info['pictures'])
        if total_images > MAX_IMAGES:
            await update.message.reply_text(
                f"⚠️ Показано {sent_images} из {total_images} изображений. "
                f"Все изображения сохранены в директории data/images."
            )
            
        # Отправляем файл Markdown с внешними ссылками на изображения
        if images_info['markdown']:
            with open(images_info['markdown'], 'rb') as doc_file:
                await update.message.reply_document(
                    document=doc_file,
                    caption=f"📝 Markdown-документ с внешними ссылками на изображения"
                )
    
    except Exception as e:
        logger.error(f"Ошибка при извлечении изображений: {e}")
        await progress_message.edit_text(
            f"❌ Произошла ошибка при извлечении изображений: {str(e)[:100]}..."
        )
        pdf_processor.is_processing = False


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений"""
    # Если бот еще не инициализирован, запускаем инициализацию
    global session_manager, pdf_processor
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
        # Проверяем, есть ли у нас активные изображения из PDF
        images_info = context.chat_data.get("last_extracted_images")
        
        if images_info:
            # Используем режим с изображениями
            result = await session_manager.send_message_with_images(user_id, user_message, images_info)
            response = result["response"]
            relevant_images = result["relevant_images"]
            
            # Отправляем ответ пользователю
            await processing_message.edit_text(response)
            
            # Отправляем релевантные изображения, если они есть
            if relevant_images:
                # Подготавливаем текст для сообщения с изображениями
                images_text = "Вот релевантные изображения к моему ответу:"
                
                # Отправляем изображения страниц
                for page_info in relevant_images.get("pages", []):
                    with open(page_info['file_path'], 'rb') as photo:
                        await update.message.reply_photo(
                            photo=photo,
                            caption=f"📄 Страница {page_info['page_no']} из документа {images_info['document_name']}"
                        )
                
                # Отправляем изображения таблиц
                for table_info in relevant_images.get("tables", []):
                    with open(table_info['file_path'], 'rb') as photo:
                        await update.message.reply_photo(
                            photo=photo,
                            caption=f"📊 Таблица {table_info['table_no']} из документа {images_info['document_name']}"
                        )
                
                # Отправляем изображения рисунков
                for picture_info in relevant_images.get("pictures", []):
                    with open(picture_info['file_path'], 'rb') as photo:
                        await update.message.reply_photo(
                            photo=photo,
                            caption=f"🖼 Рисунок {picture_info['picture_no']} из документа {images_info['document_name']}"
                        )
        else:
            # Обычный режим без изображений
            response = await session_manager.send_message(user_id, user_message)
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
    
    # Инициализация PDF процессора
    logger.info("Инициализация PDF процессора")
    pdf_processor = PDFProcessor(sdk, update_callback=update_progress)
    
    # Проверяем существование индекса
    search_index = None
    existing_index = await pdf_processor.check_existing_index()
    
    if existing_index:
        # Если существующий индекс найден, используем его
        search_index = existing_index
        if progress_message:
            await progress_message.edit_text(f"✅ Использую существующий поисковый индекс: {search_index.id}")
    else:
        # Если индекс не найден, обрабатываем документы и создаем новый
        if progress_message:
            await progress_message.edit_text("Существующий индекс не найден. Начинаю обработку PDF-файлов...")
            
        # Обработка документов
        files = await pdf_processor.convert_and_upload_pdfs()
        
        # Создание индекса
        if files:
            search_index = await pdf_processor.create_search_index()
            if not search_index and progress_message:
                await progress_message.edit_text(
                    "⚠️ Не удалось создать поисковый индекс. Бот будет работать в режиме обычного ассистента."
                )
        elif progress_message:
            await progress_message.edit_text(
                "⚠️ Нет файлов для индексации. Бот будет работать в режиме обычного ассистента."
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
    model = sdk.models.completions("yandexgpt-lite", model_version="rc")
    
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
    application.add_handler(CommandHandler("reindex", reindex_command))
    application.add_handler(CommandHandler("extract_images", extract_images_command))

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
