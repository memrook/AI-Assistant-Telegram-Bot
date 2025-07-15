import logging
import json
from typing import Dict, List
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
        self.conversation_history: Dict[int, List[Dict]] = {}  # Хранение истории сообщений для каждого пользователя
        
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
    
    async def send_message(self, user_id: int, message: str) -> str:
        """Отправляет сообщение в тред пользователя и получает ответ от ассистента"""
        thread = await self.get_user_thread(user_id)
        
        # Сохраняем сообщение пользователя в истории
        self._add_to_history(user_id, "user", message)
        
        try:
            # Записываем сообщение пользователя в тред используя стандартный метод
            thread.write(message)
            logger.info(f"Сообщение пользователя {user_id} записано в тред")
            
            # Запускаем ассистента с использованием треда
            logger.info(f"Запускаем ассистента для пользователя {user_id}")
            
            try:
                # Используем базовый запуск ассистента (совместимый с новой версией API)
                run = self.assistant.run(thread)
                logger.info("Запущен с базовыми параметрами")
                
                # Ждем результат от ассистента с улучшенной обработкой ошибок
                result = self._wait_for_result(run)
                
                if result is None:
                    # Если результат None, значит произошла ошибка при выполнении
                    error_msg = "Ассистент не смог обработать ваш запрос. Попробуйте переформулировать вопрос или повторить позже."
                    self._add_to_history(user_id, "assistant", error_msg)
                    return error_msg
                
                # Формируем ответ из результата
                response = self._format_response(result)
                
                if not response or response.strip() == "":
                    response = "Извините, не удалось получить ответ на ваш вопрос. Попробуйте переформулировать запрос."
                
                # Сохраняем ответ ассистента в истории
                self._add_to_history(user_id, "assistant", response)
                    
                logger.info(f"Получен ответ для пользователя {user_id}")
                return response
                
            except Exception as e:
                error_msg = f"Ошибка при получении ответа от ассистента: {e}"
                logger.error(error_msg)
                
                # Проверяем, содержит ли ошибка информацию о failed run
                if "failed" in str(e).lower():
                    user_error_msg = "Произошла ошибка при обработке запроса в AI модели. Попробуйте:"
                    user_error_msg += "\n• Переформулировать вопрос"
                    user_error_msg += "\n• Задать более простой вопрос"
                    user_error_msg += "\n• Повторить запрос через некоторое время"
                else:
                    user_error_msg = f"Произошла техническая ошибка при обработке запроса. Попробуйте позже."
                
                self._add_to_history(user_id, "assistant", user_error_msg)
                return user_error_msg
                
        except Exception as e:
            error_msg = f"Ошибка при записи сообщения в тред: {e}"
            logger.error(error_msg)
            return f"Произошла ошибка при отправке сообщения: {str(e)}"
            
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

 