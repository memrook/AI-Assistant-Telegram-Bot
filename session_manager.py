import logging
import json
import asyncio
import time
from typing import Dict, List, Optional
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk._assistants.assistant import Assistant
from yandex_cloud_ml_sdk._threads.thread import Thread
from database import DatabaseManager

logger = logging.getLogger(__name__)

# TODO Добавить кнопку "Повторить" при ошибке "Статус выполнения run: RunStatus.FAILED"
class SessionManager:
    """Класс для управления сессиями и тредами пользователей"""
    
    def __init__(self, sdk: YCloudML, assistant: Assistant, db_manager: DatabaseManager = None):
        self.sdk = sdk
        self.assistant = assistant
        self.user_threads: Dict[int, Thread] = {}
        self.conversation_history: Dict[int, List[Dict]] = {}  # Хранение истории сообщений для каждого пользователя
        self.db_manager = db_manager or DatabaseManager()
        self.user_db_ids: Dict[int, int] = {}  # Маппинг telegram_id -> db_user_id
        self.active_conversations: Dict[int, int] = {}  # Маппинг telegram_id -> conversation_id
    
    async def initialize_user_in_db(self, telegram_id: int, username: str = None, 
                                   first_name: str = None, last_name: str = None) -> int:
        """Инициализирует пользователя в базе данных и возвращает его db_user_id"""
        if telegram_id not in self.user_db_ids:
            db_user_id = await self.db_manager.get_or_create_user(
                telegram_id, username, first_name, last_name
            )
            self.user_db_ids[telegram_id] = db_user_id
            logger.info(f"Пользователь {telegram_id} инициализирован в БД с ID {db_user_id}")
        return self.user_db_ids[telegram_id]
    
    async def start_conversation_in_db(self, telegram_id: int) -> int:
        """Начинает новый диалог в базе данных"""
        db_user_id = self.user_db_ids.get(telegram_id)
        if not db_user_id:
            logger.warning(f"Пользователь {telegram_id} не найден в БД при начале диалога")
            return None
            
        conversation_id = await self.db_manager.start_conversation(db_user_id)
        self.active_conversations[telegram_id] = conversation_id
        logger.info(f"Начат диалог {conversation_id} для пользователя {telegram_id}")
        return conversation_id
        
    async def get_user_thread(self, user_id: int) -> Thread:
        """Получает тред пользователя или создает новый, если тред не существует"""
        if user_id not in self.user_threads:
            logger.info(f"Создаем новый тред для пользователя {user_id}")
            thread = self.sdk.threads.create()
            self.user_threads[user_id] = thread
            # Инициализируем историю сообщений для нового пользователя
            self.conversation_history[user_id] = []
            return thread
        
        return self.user_threads[user_id]
        
    def _add_to_history(self, user_id: int, role: str, content: str):
        """Добавляет сообщение в историю пользователя"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
            
        self.conversation_history[user_id].append({
            "role": role,
            "content": content
        })
        
        # Ограничиваем историю последними 20 сообщениями
        if len(self.conversation_history[user_id]) > 20:
            self.conversation_history[user_id] = self.conversation_history[user_id][-20:]
    
    async def send_message(self, user_id: int, message: str, 
                          username: str = None, first_name: str = None, last_name: str = None) -> str:
        """Отправляет сообщение в тред пользователя и получает ответ от ассистента"""
        start_time = time.time()
        
        # Инициализируем пользователя в БД если нужно
        await self.initialize_user_in_db(user_id, username, first_name, last_name)
        
        # Проверяем активный диалог или начинаем новый
        conversation_id = self.active_conversations.get(user_id)
        if not conversation_id:
            conversation_id = await self.start_conversation_in_db(user_id)
        
        thread = await self.get_user_thread(user_id)
        
        # Сохраняем сообщение пользователя в истории
        self._add_to_history(user_id, "user", message)
        
        # Сохраняем сообщение пользователя в БД
        if self.db_manager and conversation_id:
            try:
                await self.db_manager.add_message(
                    conversation_id, "user", message, "text"
                )
            except Exception as e:
                logger.error(f"Ошибка сохранения сообщения пользователя в БД: {e}")
        
        try:
            # Записываем сообщение пользователя в тред используя стандартный метод
            thread.write(message)
            logger.info(f"Сообщение пользователя {user_id} записано в тред")
            
            # Запускаем ассистента с использованием треда с retry логикой
            logger.info(f"Запускаем ассистента для пользователя {user_id}")
            
            result = await self._run_assistant_with_retry(thread, user_id)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            if result is None:
                # Если результат None, значит произошла ошибка при выполнении
                error_msg = "Ассистент не смог обработать ваш запрос. Попробуйте переформулировать вопрос или повторить позже."
                self._add_to_history(user_id, "assistant", error_msg)
                
                # Сохраняем ошибку в БД
                if self.db_manager and conversation_id:
                    try:
                        await self.db_manager.add_message(
                            conversation_id, "assistant", error_msg, "text",
                            processing_time_ms, None, True, "Assistant returned None result"
                        )
                    except Exception as e:
                        logger.error(f"Ошибка сохранения ошибочного ответа в БД: {e}")
                
                return error_msg
            
            # Формируем ответ из результата
            response = self._format_response(result)
            
            if not response or response.strip() == "":
                response = "Извините, не удалось получить ответ на ваш вопрос. Попробуйте переформулировать запрос."
            
            # Сохраняем ответ ассистента в истории
            self._add_to_history(user_id, "assistant", response)
            
            # Сохраняем ответ ассистента в БД
            if self.db_manager and conversation_id:
                try:
                    await self.db_manager.add_message(
                        conversation_id, "assistant", response, "text",
                        processing_time_ms, None, False, None
                    )
                except Exception as e:
                    logger.error(f"Ошибка сохранения ответа ассистента в БД: {e}")
                
            logger.info(f"Получен ответ для пользователя {user_id} за {processing_time_ms}мс")
            return response
                
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Ошибка при записи сообщения в тред: {e}"
            logger.error(error_msg)
            
            # Сохраняем ошибку в БД
            if self.db_manager and conversation_id:
                try:
                    await self.db_manager.add_message(
                        conversation_id, "assistant", f"Произошла ошибка при отправке сообщения: {str(e)}", 
                        "text", processing_time_ms, None, True, str(e)
                    )
                except Exception as db_e:
                    logger.error(f"Ошибка сохранения ошибки в БД: {db_e}")
            
            return f"Произошла ошибка при отправке сообщения: {str(e)}"
    
    async def _run_assistant_with_retry(self, thread, user_id: int, max_retries: int = 3):
        """Запускает ассистента с retry логикой при ошибках FAILED статуса"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Попытка {attempt + 1}/{max_retries} запуска ассистента для пользователя {user_id}")
                
                # Используем базовый запуск ассистента (совместимый с новой версией API)
                run = self.assistant.run(thread)
                logger.info(f"Run запущен (попытка {attempt + 1})")
                
                # Ждем результат от ассистента с улучшенной обработкой ошибок
                result = self._wait_for_result(run)
                
                if result is not None:
                    logger.info(f"Успешно получен результат на попытке {attempt + 1}")
                    return result
                else:
                    logger.warning(f"Попытка {attempt + 1} вернула None результат")
                    if attempt < max_retries - 1:
                        # Делаем паузу перед следующей попыткой (экспоненциальная задержка)
                        delay = 2 ** attempt  # 1s, 2s, 4s
                        logger.info(f"Ожидаем {delay} секунд перед следующей попыткой...")
                        await asyncio.sleep(delay)
                    continue
                    
            except Exception as e:
                error_msg = f"Ошибка при получении ответа от ассистента (попытка {attempt + 1}): {e}"
                logger.error(error_msg)
                
                # Проверяем, содержит ли ошибка информацию о failed run
                if "failed" in str(e).lower():
                    logger.warning(f"Обнаружена FAILED ошибка на попытке {attempt + 1}")
                    if attempt < max_retries - 1:
                        # Делаем паузу перед следующей попыткой
                        delay = 2 ** attempt  # 1s, 2s, 4s
                        logger.info(f"Ожидаем {delay} секунд перед повторной попыткой...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # Последняя попытка не удалась
                        user_error_msg = "Произошла ошибка при обработке запроса в AI модели. Попробуйте:"
                        user_error_msg += "\n• Переформулировать вопрос"
                        user_error_msg += "\n• Задать более простой вопрос"
                        user_error_msg += "\n• Повторить запрос через некоторое время"
                        self._add_to_history(user_id, "assistant", user_error_msg)
                        return None
                else:
                    # Для других ошибок не делаем retry
                    user_error_msg = f"Произошла техническая ошибка при обработке запроса. Попробуйте позже."
                    self._add_to_history(user_id, "assistant", user_error_msg)
                    return None
                    
        # Если все попытки исчерпаны
        logger.error(f"Все {max_retries} попытки исчерпаны для пользователя {user_id}")
        error_msg = f"Не удалось получить ответ после {max_retries} попыток. Попробуйте позже."
        self._add_to_history(user_id, "assistant", error_msg)
        return None
            
    def _wait_for_result(self, run):
        """Ожидает результат выполнения ассистента с расширенной обработкой ошибок"""
        try:
            # Ждем завершения run
            run_result = run.wait()
            
            # Проверяем статус выполнения, если доступен
            if hasattr(run_result, 'status'):
                logger.info(f"Статус выполнения run: {run_result.status}")
                if run_result.status == 'failed':
                    logger.error("Run завершился с ошибкой")
                    return None
            
            # Проверяем наличие результата
            if hasattr(run_result, 'message') and run_result.message is not None:
                logger.info("Получен результат через атрибут message")
                return run_result.message
            elif run_result is not None:
                logger.info("Получен результат напрямую")
                return run_result
            else:
                logger.warning("Run завершился, но результат пустой")
                return None
                
        except AttributeError as e:
            logger.error(f"Ошибка атрибута при ожидании результата: {e}")
            try:
                # Альтернативный способ получения результата
                result = run.wait()
                if result is not None:
                    return result
                else:
                    logger.error("Альтернативный способ тоже вернул None")
                    return None
            except Exception as e2:
                logger.error(f"Альтернативный способ получения результата не сработал: {e2}")
                return None
                
        except Exception as e:
            logger.error(f"Неожиданная ошибка при ожидании результата: {e}")
            
            # Проверяем, содержит ли ошибка информацию о том, что run failed
            if "failed" in str(e).lower() and "don't have a message result" in str(e).lower():
                logger.error("Run завершился с ошибкой и не содержит результата")
                return None
            
            # Для других ошибок пытаемся получить хоть какой-то результат
            try:
                return run.wait()
            except:
                return None
                
    def _format_response(self, result):
        """Форматирует ответ из результата работы ассистента"""
        if result is None:
            return "Получен пустой ответ от ассистента."
            
        try:
            # Проверяем формат результата и выбираем соответствующий метод обработки
            response = ""
            
            # Вариант 1: результат имеет атрибут parts
            if hasattr(result, 'parts') and result.parts:
                logger.info("Форматируем ответ из parts")
                for part in result.parts:
                    if hasattr(part, 'text'):
                        response += part.text
                    else:
                        response += str(part)
                        
            # Вариант 2: результат имеет атрибут text
            elif hasattr(result, 'text') and result.text:
                logger.info("Форматируем ответ из атрибута text")
                response = result.text
                
            # Вариант 3: результат имеет атрибут content
            elif hasattr(result, 'content') and result.content:
                logger.info("Форматируем ответ из атрибута content")
                response = result.content
                
            # Вариант 4: результат сам является строкой или преобразуемым к строке объектом
            elif result is not None:
                logger.info("Форматируем ответ через str()")
                response = str(result)
                
            else:
                logger.warning("Не удалось определить формат результата")
                response = "Получен ответ неизвестного формата от ассистента."
                
            # Убираем лишние пробелы и проверяем, что ответ не пустой
            response = response.strip()
            if not response:
                response = "Получен пустой ответ от ассистента."
                
            logger.info(f"Отформатированный ответ (длина: {len(response)} символов)")
            return response
            
        except Exception as e:
            logger.error(f"Ошибка при форматировании ответа: {e}")
            return "Ошибка форматирования ответа ассистента."
    
    async def reset_user_thread(self, user_id: int) -> None:
        """Сбрасывает тред пользователя, создавая новый"""
        # Завершаем текущий диалог в БД
        conversation_id = self.active_conversations.get(user_id)
        if conversation_id and self.db_manager:
            try:
                await self.db_manager.end_conversation(conversation_id)
                logger.info(f"Завершен диалог {conversation_id} для пользователя {user_id}")
            except Exception as e:
                logger.error(f"Ошибка завершения диалога в БД: {e}")
        
        # Удаляем из активных диалогов
        self.active_conversations.pop(user_id, None)
        
        if user_id in self.user_threads:
            logger.info(f"Сбрасываем тред для пользователя {user_id}")
            thread = self.sdk.threads.create()
            self.user_threads[user_id] = thread
            # Также сбрасываем историю сообщений
            self.conversation_history[user_id] = []
            
    def get_conversation_history(self, user_id: int) -> str:
        """Возвращает историю разговора пользователя в читаемом формате"""
        if user_id not in self.conversation_history or not self.conversation_history[user_id]:
            return "История разговора пуста."
            
        formatted_history = []
        for message in self.conversation_history[user_id]:
            role = "👤 Вы" if message["role"] == "user" else "🤖 Ассистент"
            formatted_history.append(f"{role}: {message['content']}")
            
        return "\n\n".join(formatted_history)
    
    async def get_user_stats_from_db(self, user_id: int) -> Optional[Dict]:
        """Получает статистику пользователя из базы данных"""
        if not self.db_manager:
            return None
        try:
            return await self.db_manager.get_user_stats(user_id)
        except Exception as e:
            logger.error(f"Ошибка получения статистики пользователя из БД: {e}")
            return None
    
    async def get_global_stats_from_db(self, days: int = 30) -> Optional[Dict]:
        """Получает глобальную статистику из базы данных"""
        if not self.db_manager:
            return None
        try:
            return await self.db_manager.get_global_stats(days)
        except Exception as e:
            logger.error(f"Ошибка получения глобальной статистики из БД: {e}")
            return None
    
    async def export_user_conversations(self, user_id: int, start_date: str = None, end_date: str = None) -> Optional[List]:
        """Экспортирует диалоги пользователя из базы данных"""
        if not self.db_manager:
            return None
        try:
            return await self.db_manager.export_conversations(user_id, start_date, end_date)
        except Exception as e:
            logger.error(f"Ошибка экспорта диалогов из БД: {e}")
            return None

 