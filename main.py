import asyncio
import logging
import signal
from dotenv import load_dotenv
from telegram_bot import main as run_bot
import time

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Переменная для хранения приложения бота
bot_app = None
# Флаг для отслеживания, выполняется ли завершение уже
is_shutting_down = False

# Обработчик сигналов для корректного завершения
def signal_handler(sig, frame):
    logger.info("Получен сигнал завершения, останавливаем приложение...")
    if asyncio.get_event_loop().is_running():
        asyncio.create_task(shutdown())

async def shutdown():
    """Корректное завершение работы бота и приложения"""
    global bot_app, is_shutting_down
    
    # Предотвращаем повторное выполнение завершения
    if is_shutting_down:
        logger.info("Завершение уже выполняется...")
        return
        
    is_shutting_down = True
    
    if bot_app:
        logger.info("Останавливаем бота...")
        try:
            # Сначала останавливаем updater, если он запущен
            if hasattr(bot_app, 'updater') and bot_app.updater:
                try:
                    logger.info("Останавливаем updater...")
                    await bot_app.updater.stop()
                    logger.info("Updater остановлен")
                except Exception as e:
                    logger.error(f"Ошибка при остановке updater: {e}")
            
            # Затем останавливаем само приложение
            try:
                logger.info("Останавливаем application...")
                await bot_app.stop()
                logger.info("Application остановлен")
            except Exception as e:
                logger.error(f"Ошибка при остановке application: {e}")
                
            # И в конце закрываем соединения
            try:
                logger.info("Закрываем соединения...")
                await bot_app.shutdown()
                logger.info("Соединения закрыты")
            except Exception as e:
                logger.error(f"Ошибка при закрытии соединений: {e}")
                
            logger.info("Бот успешно остановлен.")
        except Exception as e:
            logger.error(f"Неожиданная ошибка при остановке бота: {e}")
    else:
        logger.info("Бот не был запущен.")
        
    # Отменяем все задачи кроме текущей
    try:
        logger.info("Отменяем все оставшиеся задачи...")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        logger.info(f"Отменено {len(tasks)} задач")
    except Exception as e:
        logger.error(f"Ошибка при отмене задач: {e}")
    
    # Останавливаем цикл событий
    try:
        logger.info("Останавливаем цикл событий...")
        loop = asyncio.get_event_loop()
        loop.stop()
        logger.info("Цикл событий остановлен")
    except Exception as e:
        logger.error(f"Ошибка при остановке цикла событий: {e}")

async def main():
    """Основная функция для запуска приложения"""
    global bot_app
    logger.info("Запуск приложения AI-ассистента")
    
    # Запускаем бота
    bot_app = await run_bot()
    
    # Бесконечный цикл для поддержания работы приложения
    try:
        # Держим приложение запущенным, пока не будет прервано
        logger.info("Приложение запущено и работает. Нажмите Ctrl+C для завершения.")
        
        # Бесконечное ожидание, пока не будет прервано сигналом
        while True:
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        logger.info("Основная задача отменена")
    finally:
        # Корректное завершение
        await shutdown()

if __name__ == "__main__":
    # Регистрируем обработчики сигналов для корректного завершения
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Приложение остановлено пользователем")
    except Exception as e:
        logger.error(f"Произошла ошибка при выполнении приложения: {e}")
        # Если произошла ошибка, убедимся, что бот корректно остановлен
        try:
            # Пытаемся использовать существующий цикл событий или создаем новый
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(shutdown())
                    # Даем время, чтобы задача успела выполниться
                    time.sleep(2)
                else:
                    loop.run_until_complete(shutdown())
            except RuntimeError:
                # Если цикл событий закрыт, создаем новый
                logger.info("Создаем новый цикл событий для завершения")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(shutdown())
                loop.close()
        except Exception as cleanup_error:
            logger.error(f"Ошибка при попытке корректного завершения: {cleanup_error}")
    finally:
        logger.info("Приложение завершено")