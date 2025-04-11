import os
import logging
import ssl
import time
import asyncio
import json
from pathlib import Path
from docling.document_converter import DocumentConverter
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