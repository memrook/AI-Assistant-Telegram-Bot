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

 