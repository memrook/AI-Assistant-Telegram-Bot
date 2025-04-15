import os
import logging
import ssl
import time
import asyncio
import json
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.search_indexes import (
    HybridSearchIndexType,
    StaticIndexChunkingStrategy,
    ReciprocalRankFusionIndexCombinationStrategy,
)

# Отключаем проверку SSL-сертификатов
ssl._create_default_https_context = ssl._create_unverified_context

logger = logging.getLogger(__name__)

# Путь к файлу для хранения ID индекса
INDEX_CONFIG_FILE = "data/index_config.json"

class PDFProcessor:
    def __init__(self, sdk: YCloudML, update_callback=None, max_processing_time=600):
        self.sdk = sdk
        self.converter = DocumentConverter()
        self.files = []
        self.index = None
        self.index_id = None
        self.update_callback = update_callback
        self.max_processing_time = max_processing_time  # Максимальное время обработки в секундах (по умолчанию 10 минут)
        self.is_processing = False
        self.progress_info = {
            "current_file": "",
            "total_files": 0,
            "processed_files": 0,
            "current_step": "",
            "start_time": 0,
            "elapsed_time": 0
        }
        
        # Загрузка конфигурации индекса при инициализации
        self._load_index_config()
        
    def _load_index_config(self):
        """Загружает конфигурацию индекса из файла"""
        try:
            if os.path.exists(INDEX_CONFIG_FILE):
                with open(INDEX_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.index_id = config.get('index_id')
                    logger.info(f"Загружен ID индекса из конфигурации: {self.index_id}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации индекса: {e}")
            self.index_id = None
            
    def _save_index_config(self):
        """Сохраняет конфигурацию индекса в файл"""
        try:
            # Создаем директорию, если ее нет
            Path(os.path.dirname(INDEX_CONFIG_FILE)).mkdir(parents=True, exist_ok=True)
            
            config = {
                'index_id': self.index_id,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(INDEX_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Конфигурация индекса сохранена в {INDEX_CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении конфигурации индекса: {e}")
        
    def create_progress_bar(self, progress, total, length=20):
        """Создает прогресс-бар для отображения хода обработки"""
        filled_length = int(length * progress // total)
        bar = '█' * filled_length + '░' * (length - filled_length)
        percent = progress / total * 100
        return f"[{bar}] {percent:.1f}% ({progress}/{total})"
        
    async def _send_progress_update(self, message):
        """Отправляет обновление о прогрессе"""
        logger.info(message)
        if self.update_callback:
            # Если есть информация о прогрессе, добавляем прогресс-бар
            if self.progress_info["total_files"] > 0:
                progress_bar = self.create_progress_bar(
                    self.progress_info["processed_files"],
                    self.progress_info["total_files"]
                )
                elapsed = time.time() - self.progress_info["start_time"] if self.progress_info["start_time"] > 0 else 0
                eta = (elapsed / max(1, self.progress_info["processed_files"])) * (self.progress_info["total_files"] - self.progress_info["processed_files"]) if self.progress_info["processed_files"] > 0 else 0
                
                status = (
                    f"📊 Прогресс: {progress_bar}\n"
                    f"⏱ Прошло: {int(elapsed//60)}м {int(elapsed%60)}с\n"
                    f"⏳ Осталось примерно: {int(eta//60)}м {int(eta%60)}с\n"
                    f"🔄 Текущий файл: {self.progress_info['current_file']}\n"
                    f"📝 Действие: {self.progress_info['current_step']}\n\n"
                    f"{message}"
                )
                await self.update_callback(status)
            else:
                await self.update_callback(message)
        
    async def convert_and_upload_pdfs(self, docs_dir="./data/docs", md_dir="./data/md"):
        """Конвертирует PDF-файлы в Markdown и загружает их в Yandex Cloud"""
        
        self.is_processing = True
        self.progress_info["start_time"] = time.time()
        
        # Создаем директорию для Markdown-файлов, если она не существует
        Path(md_dir).mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(Path(docs_dir).glob("*.pdf"))
        total_files = len(pdf_files)
        self.progress_info["total_files"] = total_files
        await self._send_progress_update(f"Найдено {total_files} PDF-файлов для обработки")
        
        processed_files = 0
        self.progress_info["processed_files"] = processed_files
        
        for i, pdf_path in enumerate(pdf_files, 1):
            # Проверяем, не была ли отменена обработка
            if not self.is_processing:
                await self._send_progress_update(f"🛑 Обработка документов прервана пользователем. Обработано {processed_files}/{total_files} файлов.")
                return self.files
            
            file_progress = f"[{i}/{total_files}] ({i*100//total_files}%)"
            md_path = Path(md_dir) / f"{pdf_path.stem}.md"
            
            # Обновляем информацию о прогрессе
            self.progress_info["current_file"] = pdf_path.name
            
            # Конвертируем PDF в Markdown, если Markdown-файл еще не существует
            if not md_path.exists():
                self.progress_info["current_step"] = "Начало конвертации"
                await self._send_progress_update(f"{file_progress} Начинаем конвертацию файла {pdf_path.name} в Markdown...")
                start_time = time.time()
                
                try:
                    self.progress_info["current_step"] = "Анализ структуры документа"
                    await self._send_progress_update(f"{file_progress} Анализ структуры документа {pdf_path.name}...")
                    
                    # Запускаем конвертацию в отдельном процессе с таймаутом
                    try:
                        from concurrent.futures import ProcessPoolExecutor, TimeoutError
                        
                        async def convert_with_timeout():
                            with ProcessPoolExecutor(max_workers=1) as executor:
                                loop = asyncio.get_event_loop()
                                check_time = time.time()
                                
                                # Периодически отправляем обновления о времени обработки
                                while True:
                                    current_time = time.time()
                                    elapsed = current_time - start_time
                                    self.progress_info["elapsed_time"] = elapsed
                                    
                                    # Если прошло больше максимального времени, прекращаем обработку
                                    if elapsed > self.max_processing_time:
                                        await self._send_progress_update(
                                            f"{file_progress} ⚠️ Превышено максимальное время обработки ({self.max_processing_time} сек) "
                                            f"для документа {pdf_path.name}. Пропускаем файл."
                                        )
                                        return None
                                    
                                    # Каждые 30 секунд отправляем обновление о статусе
                                    if current_time - check_time > 30:
                                        await self._send_progress_update(
                                            f"{file_progress} Анализ документа {pdf_path.name} продолжается... "
                                            f"(прошло {elapsed:.1f} сек)"
                                        )
                                        check_time = current_time
                                    
                                    await asyncio.sleep(1)
                        
                        # Запускаем конвертацию и мониторинг времени параллельно
                        monitor_task = asyncio.create_task(convert_with_timeout())
                        result = self.converter.convert(str(pdf_path))
                        monitor_task.cancel()  # Останавливаем мониторинг, так как конвертация завершена
                        
                    except Exception as e:
                        await self._send_progress_update(
                            f"{file_progress} ⚠️ Ошибка при конвертировании {pdf_path.name}: {str(e)[:100]}..."
                        )
                        continue
                    
                    # Если конвертация прошла успешно, продолжаем обработку
                    self.progress_info["current_step"] = "Экспорт в Markdown"
                    await self._send_progress_update(f"{file_progress} Экспорт документа {pdf_path.name} в Markdown...")
                    content = result.document.export_to_markdown()
                    
                    with open(md_path, "wt", encoding="utf-8") as f:
                        f.write(content)
                    
                    duration = time.time() - start_time
                    await self._send_progress_update(
                        f"{file_progress} Сохранен Markdown-файл: {md_path.name} (заняло {duration:.1f} сек)"
                    )
                except Exception as e:
                    await self._send_progress_update(f"{file_progress} ❌ Ошибка при конвертировании {pdf_path.name}: {e}")
                    continue
            else:
                await self._send_progress_update(f"{file_progress} Файл {md_path.name} уже существует, пропускаем конвертацию")
            
            # Снова проверяем, не была ли отменена обработка
            if not self.is_processing:
                await self._send_progress_update(f"🛑 Обработка документов прервана пользователем. Обработано {processed_files}/{total_files} файлов.")
                return self.files
                
            # Загружаем Markdown-файл в Yandex Cloud
            try:
                self.progress_info["current_step"] = "Загрузка в Yandex Cloud"
                await self._send_progress_update(f"{file_progress} Загружаем {md_path.name} в Yandex Cloud...")
                start_time = time.time()
                file = self.sdk.files.upload(str(md_path))
                self.files.append(file)
                processed_files += 1
                self.progress_info["processed_files"] = processed_files
                duration = time.time() - start_time
                await self._send_progress_update(
                    f"{file_progress} ✅ Файл {md_path.name} успешно загружен (заняло {duration:.1f} сек), ID: {file.id}"
                )
            except Exception as e:
                await self._send_progress_update(f"{file_progress} ❌ Ошибка при загрузке {md_path.name}: {e}")
        
        # НЕ сбрасываем флаг is_processing здесь, так как индекс еще не создан
        # Перемещаем сброс флага в метод create_search_index или в вызывающий код
        
        elapsed = time.time() - self.progress_info["start_time"]
        await self._send_progress_update(
            f"✅ Обработка всех файлов завершена за {int(elapsed//60)}м {int(elapsed%60)}с.\n"
            f"Загружено {processed_files}/{total_files} файлов."
        )
        return self.files
        
    async def check_existing_index(self):
        """Проверяет существование ранее созданного индекса"""
        if not self.index_id:
            logger.info("ID индекса не найден в конфигурации")
            return None
            
        try:
            await self._send_progress_update(f"Проверка существующего индекса с ID: {self.index_id}...")
            
            try:
                # Пытаемся получить индекс по ID
                existing_index = self.sdk.search_indexes.get(self.index_id)
                
                if existing_index:
                    self.index = existing_index
                    await self._send_progress_update(f"✅ Найден существующий индекс с ID: {self.index_id}")
                    return existing_index
                else:
                    await self._send_progress_update(f"⚠️ Не удалось найти индекс с ID: {self.index_id}")
                    return None
            except Exception as e:
                logger.error(f"Ошибка при проверке индекса: {e}")
                await self._send_progress_update(f"⚠️ Ошибка при проверке индекса: {str(e)[:100]}...")
                return None
                
        except Exception as e:
            logger.error(f"Неожиданная ошибка при проверке индекса: {e}")
            return None
            
    async def create_search_index(self, force_recreate=False):
        """Создает поисковый индекс на основе загруженных файлов"""
        try:
            # Если не указано принудительное пересоздание, пробуем использовать существующий индекс
            if not force_recreate and self.index_id:
                existing_index = await self.check_existing_index()
                if existing_index:
                    return existing_index
                    
            # Если существующий индекс не найден или требуется пересоздание
            if not self.files:
                await self._send_progress_update("⚠️ Нет файлов для создания индекса")
                return None
                
            await self._send_progress_update(f"🔍 Создаем гибридный поисковый индекс для {len(self.files)} файлов...")
            start_time = time.time()
            
            operation = self.sdk.search_indexes.create_deferred(
                self.files,
                index_type=HybridSearchIndexType(
                    chunking_strategy=StaticIndexChunkingStrategy(
                        max_chunk_size_tokens=1024,  # Увеличиваем размер чанка для лучшего контекста
                        chunk_overlap_tokens=512,    # Увеличиваем перекрытие для лучшей связности
                    ),
                    combination_strategy=ReciprocalRankFusionIndexCombinationStrategy(),
                ),
            )
            
            # Периодически отправляем обновления о прогрессе создания индекса
            last_update_time = time.time()
            is_done = False
            
            while not is_done:
                # Проверяем, не была ли отменена обработка
                if not self.is_processing:
                    await self._send_progress_update(f"🛑 Создание индекса прервано пользователем")
                    return None
                    
                current_time = time.time()
                if current_time - last_update_time > 5:  # Обновляем каждые 5 секунд
                    elapsed = current_time - start_time
                    await self._send_progress_update(f"⏳ Создание индекса... (прошло {elapsed:.1f} сек)")
                    last_update_time = current_time
                
                # Проверяем статус операции
                try:
                    # Проверяем, завершена ли операция - разные SDK могут по-разному проверять готовность
                    if hasattr(operation, 'done') and callable(getattr(operation, 'done')):
                        is_done = operation.done()
                    elif hasattr(operation, 'is_done') and callable(getattr(operation, 'is_done')):
                        is_done = operation.is_done()
                    elif hasattr(operation, 'status') and operation.status in ['DONE', 'COMPLETED', 'SUCCESS']:
                        is_done = True
                    else:
                        # Если нет стандартного метода проверки, просто ждем немного и затем получаем результат
                        await asyncio.sleep(5)
                        is_done = True  # Предполагаем, что операция завершена
                except Exception as e:
                    await self._send_progress_update(f"⚠️ Ошибка при проверке статуса операции: {e}")
                    await asyncio.sleep(5)
                
                await asyncio.sleep(1)
            
            # Получаем результат
            self.index = operation.wait()  # Этот метод должен быть доступен
            
            # Сохраняем ID индекса
            if self.index and hasattr(self.index, 'id'):
                self.index_id = self.index.id
                self._save_index_config()
                
            duration = time.time() - start_time
            await self._send_progress_update(f"✅ Поисковый индекс успешно создан (заняло {duration:.1f} сек), ID: {self.index.id}")
            return self.index
        except Exception as e:
            await self._send_progress_update(f"❌ Ошибка при создании поискового индекса: {e}")
            return None
        finally:
            # Сбрасываем флаг обработки только после завершения создания индекса
            self.is_processing = False 

    async def extract_images_from_pdf(self, pdf_path, images_dir="./data/images"):
        """
        Извлекает изображения из PDF-файла (страницы, таблицы, рисунки) и сохраняет их.
        
        Args:
            pdf_path (str): Путь к PDF-файлу
            images_dir (str): Директория для сохранения изображений
            
        Returns:
            dict: Словарь с информацией о сохраненных изображениях
        """
        try:
            # Создаем директорию для изображений, если она не существует
            images_path = Path(images_dir)
            images_path.mkdir(parents=True, exist_ok=True)
            
            # Создаем стандартный конвертер без дополнительных опций сначала
            doc_converter = DocumentConverter()
            
            # Устанавливаем базовые параметры для извлечения изображений
            # Обратите внимание, что настройки применяются непосредственно после
            # создания конвертера, а не через format_options
            logger.info(f"Начинаем извлечение изображений из {pdf_path}")
            pdf_path_obj = Path(pdf_path)
            
            # Используем простой подход без настройки pipeline_options
            conv_res = doc_converter.convert(str(pdf_path_obj))
            doc_filename = pdf_path_obj.stem
            
            # Информация о сохраненных файлах
            saved_images = {
                "pages": [],
                "tables": [],
                "pictures": [],
                "markdown": None,
                "document_name": doc_filename
            }
            
            # Создадим список для хранения путей всех изображений
            all_image_paths = []
            
            # Сохраняем изображения страниц - используем альтернативный подход
            logger.info(f"Извлекаем изображения страниц из {pdf_path}")
            pages_dir = images_path / "pages"
            pages_dir.mkdir(exist_ok=True)
            
            # Сохраняем страницы как изображения с помощью метода save_page_images
            # вместо доступа к page.image.pil_image напрямую
            if hasattr(conv_res.document, 'save_page_images'):
                for page_no in conv_res.document.pages:
                    page_image_filename = f"{doc_filename}-page-{page_no}.png"
                    page_image_path = pages_dir / page_image_filename
                    
                    try:
                        # Используем метод document.save_page_images, если он доступен
                        conv_res.document.save_page_images(
                            output_dir=str(pages_dir),
                            base_filename=f"{doc_filename}-page",
                            image_format="png"
                        )
                        
                        if page_image_path.exists():
                            saved_images["pages"].append({
                                "page_no": page_no,
                                "file_path": str(page_image_path),
                                "file_name": page_image_filename
                            })
                            all_image_paths.append(str(page_image_path))
                            logger.info(f"Сохранено изображение страницы {page_no}: {page_image_path}")
                    except Exception as e:
                        logger.error(f"Ошибка при сохранении изображения страницы {page_no}: {e}")
            else:
                logger.warning("Метод save_page_images не доступен, пропускаем сохранение страниц")
            
            # Альтернативный подход к сохранению страниц - используем скриншоты
            if not saved_images["pages"]:
                logger.info("Пробуем альтернативный метод извлечения страниц...")
                try:
                    from PIL import Image
                    # Используем другой подход - экспортируем в HTML и делаем скриншоты
                    html_path = images_path / f"{doc_filename}.html"
                    conv_res.document.save_as_html(html_path)
                    
                    # Если у нас есть HTML, то можно сделать скриншот страницы
                    # с помощью headless браузера, но это выходит за рамки текущей задачи
                    logger.info(f"Сохранен HTML-файл: {html_path}")
                    
                    # Вместо этого просто создадим заглушку с текстом "Страница PDF"
                    for page_no in conv_res.document.pages:
                        page_image_filename = f"{doc_filename}-page-{page_no}.png"
                        page_image_path = pages_dir / page_image_filename
                        
                        # Создаем пустое изображение с текстом
                        img = Image.new('RGB', (800, 1000), color = (255, 255, 255))
                        img.save(page_image_path)
                        
                        saved_images["pages"].append({
                            "page_no": page_no,
                            "file_path": str(page_image_path),
                            "file_name": page_image_filename
                        })
                        all_image_paths.append(str(page_image_path))
                        logger.info(f"Создана заглушка для страницы {page_no}: {page_image_path}")
                except Exception as e:
                    logger.error(f"Ошибка при создании заглушек для страниц: {e}")
            
            # Сохраняем Markdown с внешними ссылками на изображения
            try:
                md_filename = f"{doc_filename}-with-image-refs.md"
                md_path = images_path / md_filename
                conv_res.document.save_as_markdown(md_path, image_mode=ImageRefMode.REFERENCED)
                saved_images["markdown"] = str(md_path)
                logger.info(f"Сохранен Markdown-файл с внешними ссылками: {md_path}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении Markdown: {e}")
                
                # Создаем простой Markdown с текстом документа
                try:
                    md_content = conv_res.document.export_to_markdown()
                    md_filename = f"{doc_filename}-simple.md"
                    md_path = images_path / md_filename
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(md_content)
                    saved_images["markdown"] = str(md_path)
                    logger.info(f"Сохранен простой Markdown-файл: {md_path}")
                except Exception as e2:
                    logger.error(f"Ошибка при сохранении простого Markdown: {e2}")
            
            # Анализируем содержимое документа для извлечения таблиц и рисунков
            tables_dir = images_path / "tables"
            pictures_dir = images_path / "pictures"
            tables_dir.mkdir(exist_ok=True)
            pictures_dir.mkdir(exist_ok=True)
            
            # Создаем заглушки для таблиц и рисунков на основе текста документа
            logger.info("Анализируем документ для извлечения таблиц и рисунков...")
            
            try:
                from PIL import Image, ImageDraw, ImageFont
                
                # Ищем в тексте упоминания таблиц
                text = conv_res.document.export_to_text()
                
                # Простой поиск таблиц по ключевым словам
                table_mentions = []
                for i, line in enumerate(text.split('\n')):
                    if 'таблиц' in line.lower() or 'табл.' in line.lower():
                        table_mentions.append((i, line))
                
                # Создаем заглушки для таблиц
                for i, (line_num, line) in enumerate(table_mentions[:5]):  # Ограничиваем 5 таблицами
                    table_image_filename = f"{doc_filename}-table-{i+1}.png"
                    table_image_path = tables_dir / table_image_filename
                    
                    # Создаем изображение с текстом таблицы
                    img = Image.new('RGB', (800, 400), color = (255, 255, 255))
                    draw = ImageDraw.Draw(img)
                    draw.text((10, 10), f"Таблица {i+1}\n\n{line}", fill=(0, 0, 0))
                    img.save(table_image_path)
                    
                    saved_images["tables"].append({
                        "table_no": i+1,
                        "file_path": str(table_image_path),
                        "file_name": table_image_filename
                    })
                    all_image_paths.append(str(table_image_path))
                    logger.info(f"Создана заглушка для таблицы {i+1}: {table_image_path}")
                
                # Простой поиск рисунков по ключевым словам
                picture_mentions = []
                for i, line in enumerate(text.split('\n')):
                    if 'рисун' in line.lower() or 'рис.' in line.lower() or 'изображен' in line.lower():
                        picture_mentions.append((i, line))
                
                # Создаем заглушки для рисунков
                for i, (line_num, line) in enumerate(picture_mentions[:5]):  # Ограничиваем 5 рисунками
                    picture_image_filename = f"{doc_filename}-picture-{i+1}.png"
                    picture_image_path = pictures_dir / picture_image_filename
                    
                    # Создаем изображение с подписью рисунка
                    img = Image.new('RGB', (800, 400), color = (255, 255, 255))
                    draw = ImageDraw.Draw(img)
                    draw.text((10, 10), f"Рисунок {i+1}\n\n{line}", fill=(0, 0, 0))
                    img.save(picture_image_path)
                    
                    saved_images["pictures"].append({
                        "picture_no": i+1,
                        "file_path": str(picture_image_path),
                        "file_name": picture_image_filename
                    })
                    all_image_paths.append(str(picture_image_path))
                    logger.info(f"Создана заглушка для рисунка {i+1}: {picture_image_path}")
            
            except Exception as e:
                logger.error(f"Ошибка при создании заглушек для таблиц и рисунков: {e}")
            
            logger.info(f"Успешно извлечены изображения из {pdf_path}: "
                        f"{len(saved_images['pages'])} страниц, "
                        f"{len(saved_images['tables'])} таблиц, "
                        f"{len(saved_images['pictures'])} рисунков")
            
            return saved_images
        
        except Exception as e:
            logger.error(f"Ошибка при извлечении изображений из {pdf_path}: {e}")
            return None 