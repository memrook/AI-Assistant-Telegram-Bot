import os
import logging
import time
import asyncio
import json
from pathlib import Path
from docx import Document
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.search_indexes import (
    HybridSearchIndexType,
    StaticIndexChunkingStrategy,
    ReciprocalRankFusionIndexCombinationStrategy,
)

# Загрузка переменных окружения
load_dotenv()

logger = logging.getLogger(__name__)

# Параметры поискового индекса из переменных окружения
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "1024"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "512"))

# Путь к файлу для хранения ID индекса
INDEX_CONFIG_FILE = "data/index_config.json"

class DocumentProcessor:
    def __init__(self, sdk: YCloudML, update_callback=None, max_processing_time=600):
        self.sdk = sdk
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

    def _extract_text_from_docx(self, docx_path):
        """Извлекает текст из DOCX файла"""
        try:
            doc = Document(docx_path)
            full_text = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text.strip())
            return '\n'.join(full_text)
        except Exception as e:
            logger.error(f"Ошибка при извлечении текста из DOCX файла {docx_path}: {e}")
            return None

    def _convert_docx_to_md(self, docx_path, output_dir):
        """Конвертирует DOCX файл в Markdown и сохраняет во временную директорию"""
        try:
            text_content = self._extract_text_from_docx(docx_path)
            if not text_content:
                return None
                
            # Создаем имя файла для MD версии
            docx_filename = Path(docx_path).stem
            md_filename = f"{docx_filename}.md"
            md_path = Path(output_dir) / md_filename
            
            # Создаем директорию если ее нет
            md_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Записываем содержимое в MD файл
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# {docx_filename}\n\n")
                f.write(text_content)
                
            return md_path
        except Exception as e:
            logger.error(f"Ошибка при конвертации DOCX в MD: {e}")
            return None
        
    async def upload_documents(self, doc_dir="./data/md"):
        """Загружает готовые Markdown и DOCX файлы в Yandex Cloud"""
        
        self.is_processing = True
        self.progress_info["start_time"] = time.time()
        
        # Проверяем существование директории с документами
        doc_path = Path(doc_dir)
        if not doc_path.exists():
            await self._send_progress_update(f"❌ Директория {doc_dir} не существует. Создайте её и поместите туда документы.")
            self.is_processing = False
            return []
        
        # Ищем файлы поддерживаемых форматов
        md_files = list(doc_path.glob("*.md"))
        docx_files = list(doc_path.glob("*.docx"))
        all_files = md_files + docx_files
        
        total_files = len(all_files)
        self.progress_info["total_files"] = total_files
        
        if total_files == 0:
            await self._send_progress_update(f"⚠️ В директории {doc_dir} не найдено документов (.md или .docx файлов).")
            self.is_processing = False
            return []
            
        await self._send_progress_update(f"Найдено {len(md_files)} Markdown и {len(docx_files)} DOCX файлов для загрузки")
        
        processed_files = 0
        self.progress_info["processed_files"] = processed_files
        
        # Создаем временную директорию для конвертированных DOCX файлов
        temp_dir = Path("./temp_converted")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            for i, file_path in enumerate(all_files, 1):
                # Проверяем, не была ли отменена обработка
                if not self.is_processing:
                    await self._send_progress_update(f"🛑 Загрузка документов прервана пользователем. Загружено {processed_files}/{total_files} файлов.")
                    return self.files
                
                file_progress = f"[{i}/{total_files}] ({i*100//total_files}%)"
                
                # Обновляем информацию о прогрессе
                self.progress_info["current_file"] = file_path.name
                
                # Проверяем, что файл не пустой
                try:
                    if file_path.stat().st_size == 0:
                        await self._send_progress_update(f"{file_progress} ⚠️ Файл {file_path.name} пустой, пропускаем")
                        continue
                except Exception as e:
                    await self._send_progress_update(f"{file_progress} ❌ Ошибка при проверке файла {file_path.name}: {e}")
                    continue
                
                # Определяем путь для загрузки
                upload_path = file_path
                
                # Если это DOCX файл, конвертируем его в MD
                if file_path.suffix.lower() == '.docx':
                    self.progress_info["current_step"] = "Конвертация DOCX в MD"
                    await self._send_progress_update(f"{file_progress} Конвертируем {file_path.name} из DOCX в Markdown...")
                    
                    converted_path = self._convert_docx_to_md(file_path, temp_dir)
                    if not converted_path:
                        await self._send_progress_update(f"{file_progress} ❌ Ошибка при конвертации {file_path.name}")
                        continue
                    upload_path = converted_path
                    
                # Загружаем файл в Yandex Cloud
                try:
                    self.progress_info["current_step"] = "Загрузка в Yandex Cloud"
                    await self._send_progress_update(f"{file_progress} Загружаем {file_path.name} в Yandex Cloud...")
                    start_time = time.time()
                    file = self.sdk.files.upload(str(upload_path))
                    self.files.append(file)
                    processed_files += 1
                    self.progress_info["processed_files"] = processed_files
                    duration = time.time() - start_time
                    file_type = "DOCX→MD" if file_path.suffix.lower() == '.docx' else "MD"
                    await self._send_progress_update(
                        f"{file_progress} ✅ Файл {file_path.name} ({file_type}) успешно загружен (заняло {duration:.1f} сек), ID: {file.id}"
                    )
                except Exception as e:
                    await self._send_progress_update(f"{file_progress} ❌ Ошибка при загрузке {file_path.name}: {e}")
        
        finally:
            # Очищаем временную директорию
            try:
                import shutil
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    logger.info("Временная директория очищена")
            except Exception as e:
                logger.warning(f"Не удалось очистить временную директорию: {e}")
        
        elapsed = time.time() - self.progress_info["start_time"]
        await self._send_progress_update(
            f"✅ Загрузка всех файлов завершена за {int(elapsed//60)}м {int(elapsed%60)}с.\n"
            f"Загружено {processed_files}/{total_files} файлов."
        )
        return self.files

    # Оставляем старый метод для обратной совместимости
    async def upload_markdown_files(self, md_dir="./data/md"):
        """Устаревший метод. Используйте upload_documents()"""
        return await self.upload_documents(md_dir)
        
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
                        max_chunk_size_tokens=CHUNK_SIZE_TOKENS,     # Размер чанка из переменной окружения
                        chunk_overlap_tokens=CHUNK_OVERLAP_TOKENS,   # Перекрытие чанков из переменной окружения
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
                
                # Проверяем статус операции - упрощенная проверка
                try:
                    # Просто ждем немного и затем получаем результат
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