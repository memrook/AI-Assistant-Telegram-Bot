import os
import logging
import time
import asyncio
import json
from pathlib import Path
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.search_indexes import (
    HybridSearchIndexType,
    StaticIndexChunkingStrategy,
    ReciprocalRankFusionIndexCombinationStrategy,
)

logger = logging.getLogger(__name__)

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
        
    async def upload_markdown_files(self, md_dir="./data/md"):
        """Загружает готовые Markdown-файлы в Yandex Cloud"""
        
        self.is_processing = True
        self.progress_info["start_time"] = time.time()
        
        # Проверяем существование директории с Markdown-файлами
        md_path = Path(md_dir)
        if not md_path.exists():
            await self._send_progress_update(f"❌ Директория {md_dir} не существует. Создайте её и поместите туда Markdown-файлы.")
            self.is_processing = False
            return []
        
        md_files = list(md_path.glob("*.md"))
        total_files = len(md_files)
        self.progress_info["total_files"] = total_files
        
        if total_files == 0:
            await self._send_progress_update(f"⚠️ В директории {md_dir} не найдено Markdown-файлов.")
            self.is_processing = False
            return []
            
        await self._send_progress_update(f"Найдено {total_files} Markdown-файлов для загрузки")
        
        processed_files = 0
        self.progress_info["processed_files"] = processed_files
        
        for i, md_file_path in enumerate(md_files, 1):
            # Проверяем, не была ли отменена обработка
            if not self.is_processing:
                await self._send_progress_update(f"🛑 Загрузка документов прервана пользователем. Загружено {processed_files}/{total_files} файлов.")
                return self.files
            
            file_progress = f"[{i}/{total_files}] ({i*100//total_files}%)"
            
            # Обновляем информацию о прогрессе
            self.progress_info["current_file"] = md_file_path.name
            self.progress_info["current_step"] = "Загрузка в Yandex Cloud"
            
            # Проверяем, что файл не пустой
            try:
                if md_file_path.stat().st_size == 0:
                    await self._send_progress_update(f"{file_progress} ⚠️ Файл {md_file_path.name} пустой, пропускаем")
                    continue
            except Exception as e:
                await self._send_progress_update(f"{file_progress} ❌ Ошибка при проверке файла {md_file_path.name}: {e}")
                continue
                
            # Загружаем Markdown-файл в Yandex Cloud
            try:
                await self._send_progress_update(f"{file_progress} Загружаем {md_file_path.name} в Yandex Cloud...")
                start_time = time.time()
                file = self.sdk.files.upload(str(md_file_path))
                self.files.append(file)
                processed_files += 1
                self.progress_info["processed_files"] = processed_files
                duration = time.time() - start_time
                await self._send_progress_update(
                    f"{file_progress} ✅ Файл {md_file_path.name} успешно загружен (заняло {duration:.1f} сек), ID: {file.id}"
                )
            except Exception as e:
                await self._send_progress_update(f"{file_progress} ❌ Ошибка при загрузке {md_file_path.name}: {e}")
        
        elapsed = time.time() - self.progress_info["start_time"]
        await self._send_progress_update(
            f"✅ Загрузка всех файлов завершена за {int(elapsed//60)}м {int(elapsed%60)}с.\n"
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