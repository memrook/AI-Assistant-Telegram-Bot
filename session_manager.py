import logging
import json
import re
import os
from pathlib import Path
from typing import Dict, Optional, List
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk._assistants.assistant import Assistant
from yandex_cloud_ml_sdk._threads.thread import Thread

logger = logging.getLogger(__name__)

class SessionManager:
    """Класс для управления сессиями и тредами пользователей"""
    
    def __init__(self, sdk: YCloudML, assistant: Assistant):
        self.sdk = sdk
        self.assistant = assistant
        self.user_threads: Dict[int, Thread] = {}
        self.message_history: Dict[int, List[Dict]] = {}  # Хранение истории сообщений для каждого пользователя
        
    async def get_user_thread(self, user_id: int) -> Thread:
        """Получает тред пользователя или создает новый, если тред не существует"""
        if user_id not in self.user_threads:
            logger.info(f"Создаем новый тред для пользователя {user_id}")
            thread = self.sdk.threads.create()
            self.user_threads[user_id] = thread
            # Инициализируем историю сообщений для нового пользователя
            self.message_history[user_id] = []
            return thread
        
        return self.user_threads[user_id]
        
    def _add_to_history(self, user_id: int, role: str, content: str):
        """Добавляет сообщение в историю пользователя"""
        if user_id not in self.message_history:
            self.message_history[user_id] = []
            
        self.message_history[user_id].append({
            "role": role,
            "content": content
        })
        
        # Ограничиваем историю последними 20 сообщениями
        if len(self.message_history[user_id]) > 20:
            self.message_history[user_id] = self.message_history[user_id][-20:]
    
    async def send_message(self, user_id: int, message: str) -> str:
        """Отправляет сообщение в тред пользователя и получает ответ от ассистента"""
        thread = await self.get_user_thread(user_id)
        
        # Сохраняем сообщение пользователя в истории
        self._add_to_history(user_id, "user", message)
        
        try:
            # Проверяем, поддерживает ли тред метод записи отдельных сообщений
            # Это может различаться в разных версиях SDK
            has_write_method = hasattr(thread, 'write') and callable(getattr(thread, 'write'))
            has_add_message_method = hasattr(thread, 'add_message') and callable(getattr(thread, 'add_message'))
            has_add_user_message_method = hasattr(thread, 'add_user_message') and callable(getattr(thread, 'add_user_message'))
            
            # Записываем сообщение пользователя в тред
            if has_write_method:
                # Стандартный метод в большинстве версий SDK
                thread.write(message)
                logger.info(f"Сообщение пользователя {user_id} записано в тред через write()")
            elif has_add_user_message_method:
                # Альтернативный метод в некоторых версиях
                thread.add_user_message(message)
                logger.info(f"Сообщение пользователя {user_id} записано в тред через add_user_message()")
            elif has_add_message_method:
                # Еще один альтернативный метод
                thread.add_message({"role": "user", "content": message})
                logger.info(f"Сообщение пользователя {user_id} записано в тред через add_message()")
            else:
                # Если нет подходящего метода, логируем ошибку
                logger.error("Не найден подходящий метод для записи сообщения в тред")
                return "Ошибка: не удалось отправить сообщение (не поддерживается API)"
            
            # Запускаем ассистента с использованием треда
            logger.info(f"Запускаем ассистента для пользователя {user_id}")
            
            try:
                # Пробуем запустить с различными параметрами
                run = None
                
                # Вариант 1: custom_prompt_truncation_options
                try:
                    run = self.assistant.run(
                        thread,
                        custom_prompt_truncation_options={
                            "max_messages": 15  # Увеличиваем лимит для лучшего сохранения контекста
                        }
                    )
                except (TypeError, AttributeError) as e:
                    logger.warning(f"Не удалось использовать custom_prompt_truncation_options: {e}")
                    
                    # Вариант 2: run_with_options
                    if run is None and hasattr(self.assistant, 'run_with_options'):
                        try:
                            run = self.assistant.run_with_options(thread, max_messages=15)
                        except Exception as e2:
                            logger.warning(f"Не удалось использовать run_with_options: {e2}")
                
                    # Вариант 3: run с передачей истории сообщений напрямую
                    if run is None and hasattr(self.assistant, 'run') and len(self.message_history.get(user_id, [])) > 0:
                        try:
                            messages = self.message_history[user_id]
                            run = self.assistant.run(thread, messages=messages)
                            logger.info(f"Запущен с явной передачей {len(messages)} сообщений из истории")
                        except Exception as e3:
                            logger.warning(f"Не удалось использовать явную передачу истории: {e3}")
                            
                    # Вариант 4: базовый запуск
                    if run is None:
                        run = self.assistant.run(thread)
                        logger.info("Запущен с базовыми параметрами")
                
                # Ждем результат от ассистента
                result = self._wait_for_result(run)
                
                # Формируем ответ из результата
                response = self._format_response(result)
                
                # Сохраняем ответ ассистента в истории
                self._add_to_history(user_id, "assistant", response)
                    
                logger.info(f"Получен ответ для пользователя {user_id}")
                return response
            except Exception as e:
                logger.error(f"Ошибка при получении ответа от ассистента: {e}")
                return f"Произошла ошибка при обработке запроса: {str(e)}"
        except Exception as e:
            logger.error(f"Ошибка при записи сообщения в тред: {e}")
            return f"Произошла ошибка при отправке сообщения: {str(e)}"
            
    def _wait_for_result(self, run):
        """Ожидает результат выполнения ассистента с обработкой различных форматов ответа"""
        try:
            # Вариант 1: результат возвращается как message в объекте run.wait()
            result = run.wait().message
            return result
        except (AttributeError, TypeError):
            try:
                # Вариант 2: результат возвращается напрямую из run.wait()
                result = run.wait()
                return result
            except Exception as e:
                logger.error(f"Ошибка при ожидании результата: {e}")
                raise
                
    def _format_response(self, result):
        """Форматирует ответ из результата работы ассистента"""
        try:
            # Проверяем формат результата и выбираем соответствующий метод обработки
            response = ""
            
            # Вариант 1: результат имеет атрибут parts
            if hasattr(result, 'parts'):
                for part in result.parts:
                    response += str(part)
            # Вариант 2: результат сам является строкой или преобразуемым к строке объектом
            elif result is not None:
                response = str(result)
            # Вариант 3: результат имеет атрибут text
            elif hasattr(result, 'text'):
                response = result.text
            # Вариант 4: результат имеет атрибут content
            elif hasattr(result, 'content'):
                response = result.content
            else:
                response = "Получен пустой ответ от ассистента."
            
            # Проверяем, есть ли упоминания изображений в ответе и добавляем теги изображений
            response = self._enhance_response_with_images(response)
                
            return response
        except Exception as e:
            logger.error(f"Ошибка при форматировании ответа: {e}")
            return "Ошибка форматирования ответа ассистента."
    
    def _enhance_response_with_images(self, response):
        """Улучшает ответ, добавляя ссылки на изображения, если они упоминаются"""
        try:
            # Ищем упоминания PDF файлов
            pdf_mentions = re.findall(r'документ[а-яё\s]*[«"\'](.*?)[»"\']', response, re.IGNORECASE)
            pdf_mentions += re.findall(r'файл[а-яё\s]*[«"\'](.*?)[»"\']', response, re.IGNORECASE)
            pdf_mentions += re.findall(r'из[а-яё\s]*[«"\'](.*?)[»"\']', response, re.IGNORECASE)
            
            if not pdf_mentions:
                return response
                
            # Проверяем наличие изображений для упомянутых PDF
            img_dir = Path('./data/images')
            if not img_dir.exists():
                return response
                
            # Ищем изображения, соответствующие упомянутым PDF
            added_images = []
            images_metadata = {}
            
            for pdf_name in pdf_mentions:
                # Нормализуем имя PDF (убираем расширение, если есть)
                pdf_stem = pdf_name.split('.')[0].strip().lower()
                
                # Ищем изображения, соответствующие этому PDF
                matching_images = list(img_dir.glob(f"{pdf_stem}*.jpeg")) + list(img_dir.glob(f"{pdf_stem}*.jpg")) + list(img_dir.glob(f"{pdf_stem}*.png"))
                
                # Берем до 3 изображений для каждого PDF
                for img_path in matching_images[:3]:
                    if str(img_path) not in added_images:
                        # Добавляем тег изображения в текст
                        img_tag = f"\n\n![Изображение из документа {pdf_name}]({img_path})\n"
                        response += img_tag
                        added_images.append(str(img_path))
                        
                        # Сохраняем метаданные об изображениях
                        if pdf_stem not in images_metadata:
                            images_metadata[pdf_stem] = []
                        images_metadata[pdf_stem].append(str(img_path))
            
            # Добавляем метаданные об изображениях в комментарий
            if images_metadata:
                response += f"\n\n<!-- IMAGES: {json.dumps(images_metadata, ensure_ascii=False)} -->"
                
            return response
        except Exception as e:
            logger.error(f"Ошибка при улучшении ответа изображениями: {e}")
            return response
    
    async def reset_user_thread(self, user_id: int) -> None:
        """Сбрасывает тред пользователя, создавая новый"""
        if user_id in self.user_threads:
            logger.info(f"Сбрасываем тред для пользователя {user_id}")
            thread = self.sdk.threads.create()
            self.user_threads[user_id] = thread
            # Также сбрасываем историю сообщений
            self.message_history[user_id] = []
            
    def get_conversation_history(self, user_id: int) -> str:
        """Возвращает историю разговора пользователя в читаемом формате"""
        if user_id not in self.message_history or not self.message_history[user_id]:
            return "История разговора пуста."
            
        formatted_history = []
        for message in self.message_history[user_id]:
            role = "👤 Вы" if message["role"] == "user" else "🤖 Ассистент"
            formatted_history.append(f"{role}: {message['content']}")
            
        return "\n\n".join(formatted_history) 