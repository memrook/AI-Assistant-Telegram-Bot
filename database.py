import aiosqlite
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Класс для управления SQLite базой данных с диалогами бота"""
    
    def __init__(self, db_path: str = "./data/bot_analytics.db"):
        self.db_path = db_path
        # Создаем директорию для БД если её нет
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
    async def init_database(self):
        """Инициализация базы данных и создание таблиц"""
        logger.info("Инициализация базы данных")
        
        async with aiosqlite.connect(self.db_path) as db:
            # Таблица пользователей
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_id INTEGER UNIQUE NOT NULL,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_messages INTEGER DEFAULT 0,
                    total_conversations INTEGER DEFAULT 0
                )
            """)
            
            # Таблица диалогов/сессий
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ended_at TIMESTAMP NULL,
                    status TEXT DEFAULT 'active',
                    total_messages INTEGER DEFAULT 0,
                    duration_seconds INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Таблица сообщений
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_type TEXT DEFAULT 'text',
                    processing_time_ms INTEGER NULL,
                    tokens_used INTEGER NULL,
                    has_error BOOLEAN DEFAULT 0,
                    error_details TEXT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            """)
            
            # Индексы для производительности
            await db.execute("CREATE INDEX IF NOT EXISTS idx_users_telegram_id ON users (telegram_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations (user_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages (conversation_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages (timestamp)")
            
            await db.commit()
            logger.info("База данных успешно инициализирована")
    
    async def get_or_create_user(self, telegram_id: int, username: str = None, 
                                first_name: str = None, last_name: str = None) -> int:
        """Получает или создает пользователя, возвращает user_id"""
        async with aiosqlite.connect(self.db_path) as db:
            # Сначала пытаемся найти существующего пользователя
            cursor = await db.execute(
                "SELECT id FROM users WHERE telegram_id = ?", 
                (telegram_id,)
            )
            user = await cursor.fetchone()
            
            if user:
                # Обновляем время последней активности
                await db.execute(
                    "UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE telegram_id = ?",
                    (telegram_id,)
                )
                await db.commit()
                return user[0]
            else:
                # Создаем нового пользователя
                cursor = await db.execute("""
                    INSERT INTO users (telegram_id, username, first_name, last_name)
                    VALUES (?, ?, ?, ?)
                """, (telegram_id, username, first_name, last_name))
                await db.commit()
                logger.info(f"Создан новый пользователь: telegram_id={telegram_id}")
                return cursor.lastrowid
    
    async def start_conversation(self, user_id: int) -> int:
        """Начинает новый диалог для пользователя"""
        async with aiosqlite.connect(self.db_path) as db:
            # Закрываем предыдущий активный диалог если есть
            await db.execute("""
                UPDATE conversations 
                SET ended_at = CURRENT_TIMESTAMP, status = 'completed'
                WHERE user_id = ? AND status = 'active'
            """, (user_id,))
            
            # Создаем новый диалог
            cursor = await db.execute("""
                INSERT INTO conversations (user_id)
                VALUES (?)
            """, (user_id,))
            
            # Обновляем счетчик диалогов у пользователя
            await db.execute("""
                UPDATE users 
                SET total_conversations = total_conversations + 1
                WHERE id = ?
            """, (user_id,))
            
            await db.commit()
            logger.info(f"Начат новый диалог для пользователя {user_id}")
            return cursor.lastrowid
    
    async def get_active_conversation(self, user_id: int) -> Optional[int]:
        """Получает ID активного диалога пользователя"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT id FROM conversations 
                WHERE user_id = ? AND status = 'active'
                ORDER BY started_at DESC LIMIT 1
            """, (user_id,))
            result = await cursor.fetchone()
            return result[0] if result else None
    
    async def add_message(self, conversation_id: int, role: str, content: str,
                         message_type: str = 'text', processing_time_ms: int = None,
                         tokens_used: int = None, has_error: bool = False,
                         error_details: str = None) -> int:
        """Добавляет сообщение в диалог"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO messages 
                (conversation_id, role, content, message_type, processing_time_ms, 
                 tokens_used, has_error, error_details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (conversation_id, role, content, message_type, processing_time_ms,
                  tokens_used, has_error, error_details))
            
            # Обновляем счетчик сообщений в диалоге
            await db.execute("""
                UPDATE conversations 
                SET total_messages = total_messages + 1
                WHERE id = ?
            """, (conversation_id,))
            
            # Обновляем счетчик сообщений пользователя
            await db.execute("""
                UPDATE users 
                SET total_messages = total_messages + 1, last_active = CURRENT_TIMESTAMP
                WHERE id = (SELECT user_id FROM conversations WHERE id = ?)
            """, (conversation_id,))
            
            await db.commit()
            return cursor.lastrowid
    
    async def end_conversation(self, conversation_id: int):
        """Завершает диалог"""
        async with aiosqlite.connect(self.db_path) as db:
            # Вычисляем продолжительность диалога
            await db.execute("""
                UPDATE conversations 
                SET ended_at = CURRENT_TIMESTAMP,
                    status = 'completed',
                    duration_seconds = CAST(
                        (julianday(CURRENT_TIMESTAMP) - julianday(started_at)) * 86400 AS INTEGER
                    )
                WHERE id = ?
            """, (conversation_id,))
            await db.commit()
            logger.info(f"Диалог {conversation_id} завершен")
    
    async def get_user_stats(self, telegram_id: int) -> Dict[str, Any]:
        """Получает статистику пользователя"""
        async with aiosqlite.connect(self.db_path) as db:
            # Основная информация о пользователе
            cursor = await db.execute("""
                SELECT total_messages, total_conversations, created_at, last_active
                FROM users WHERE telegram_id = ?
            """, (telegram_id,))
            user_data = await cursor.fetchone()
            
            if not user_data:
                return {}
            
            # Статистика по дням активности (последние 30 дней)
            cursor = await db.execute("""
                SELECT DATE(m.timestamp) as day, COUNT(*) as messages_count
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                JOIN users u ON c.user_id = u.id
                WHERE u.telegram_id = ? 
                AND m.timestamp >= datetime('now', '-30 days')
                GROUP BY DATE(m.timestamp)
                ORDER BY day DESC
            """, (telegram_id,))
            daily_activity = await cursor.fetchall()
            
            # Средняя длина диалогов
            cursor = await db.execute("""
                SELECT AVG(c.total_messages) as avg_messages, AVG(c.duration_seconds) as avg_duration
                FROM conversations c
                JOIN users u ON c.user_id = u.id
                WHERE u.telegram_id = ? AND c.status = 'completed'
            """, (telegram_id,))
            avg_stats = await cursor.fetchone()
            
            return {
                'total_messages': user_data[0],
                'total_conversations': user_data[1],
                'created_at': user_data[2],
                'last_active': user_data[3],
                'daily_activity': [{'day': row[0], 'messages': row[1]} for row in daily_activity],
                'avg_messages_per_conversation': round(avg_stats[0] or 0, 2),
                'avg_conversation_duration_minutes': round((avg_stats[1] or 0) / 60, 2)
            }
    
    async def get_global_stats(self, days: int = 30) -> Dict[str, Any]:
        """Получает глобальную статистику бота"""
        async with aiosqlite.connect(self.db_path) as db:
            # Общие показатели
            cursor = await db.execute("""
                SELECT 
                    COUNT(DISTINCT u.telegram_id) as total_users,
                    COUNT(DISTINCT c.id) as total_conversations,
                    COUNT(m.id) as total_messages,
                    COUNT(CASE WHEN m.has_error = 1 THEN 1 END) as error_messages,
                    AVG(m.processing_time_ms) as avg_processing_time
                FROM users u
                LEFT JOIN conversations c ON u.id = c.user_id AND c.started_at >= datetime('now', '-' || ? || ' days')
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE u.created_at >= datetime('now', '-' || ? || ' days') OR c.started_at >= datetime('now', '-' || ? || ' days')
            """, (days, days, days))
            main_stats = await cursor.fetchone()
            
            # Активность по дням
            cursor = await db.execute("""
                SELECT 
                    DATE(m.timestamp) as day,
                    COUNT(DISTINCT c.user_id) as active_users,
                    COUNT(DISTINCT c.id) as conversations,
                    COUNT(m.id) as messages
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.timestamp >= datetime('now', '-' || ? || ' days')
                GROUP BY DATE(m.timestamp)
                ORDER BY day DESC
            """, (days,))
            daily_stats = await cursor.fetchall()
            
            # Топ часов активности
            cursor = await db.execute("""
                SELECT 
                    strftime('%H', m.timestamp) as hour,
                    COUNT(m.id) as messages_count
                FROM messages m
                WHERE m.timestamp >= datetime('now', '-' || ? || ' days')
                GROUP BY strftime('%H', m.timestamp)
                ORDER BY messages_count DESC
            """, (days,))
            hourly_stats = await cursor.fetchall()
            
            return {
                'period_days': days,
                'total_users': main_stats[0] or 0,
                'total_conversations': main_stats[1] or 0,
                'total_messages': main_stats[2] or 0,
                'error_rate': round((main_stats[3] or 0) / max(main_stats[2] or 1, 1) * 100, 2),
                'avg_processing_time_ms': round(main_stats[4] or 0, 2),
                'daily_activity': [
                    {
                        'day': row[0], 
                        'active_users': row[1], 
                        'conversations': row[2], 
                        'messages': row[3]
                    } for row in daily_stats
                ],
                'hourly_distribution': [
                    {'hour': int(row[0]), 'messages': row[1]} for row in hourly_stats
                ]
            }
    
    async def export_conversations(self, telegram_id: int = None, 
                                  start_date: str = None, end_date: str = None) -> List[Dict]:
        """Экспортирует диалоги в формате JSON"""
        async with aiosqlite.connect(self.db_path) as db:
            where_conditions = []
            params = []
            
            if telegram_id:
                where_conditions.append("u.telegram_id = ?")
                params.append(telegram_id)
            
            if start_date:
                where_conditions.append("c.started_at >= ?")
                params.append(start_date)
                
            if end_date:
                where_conditions.append("c.started_at <= ?")
                params.append(end_date)
            
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            cursor = await db.execute(f"""
                SELECT 
                    c.id as conversation_id,
                    u.telegram_id,
                    u.username,
                    u.first_name,
                    c.started_at,
                    c.ended_at,
                    c.total_messages,
                    c.duration_seconds
                FROM conversations c
                JOIN users u ON c.user_id = u.id
                {where_clause}
                ORDER BY c.started_at DESC
            """, params)
            
            conversations = []
            for row in await cursor.fetchall():
                conv_id = row[0]
                
                # Получаем сообщения диалога
                msg_cursor = await db.execute("""
                    SELECT role, content, timestamp, message_type, processing_time_ms, has_error
                    FROM messages 
                    WHERE conversation_id = ?
                    ORDER BY timestamp ASC
                """, (conv_id,))
                
                messages = [
                    {
                        'role': msg[0],
                        'content': msg[1],
                        'timestamp': msg[2],
                        'type': msg[3],
                        'processing_time_ms': msg[4],
                        'has_error': bool(msg[5])
                    } for msg in await msg_cursor.fetchall()
                ]
                
                conversations.append({
                    'conversation_id': conv_id,
                    'user': {
                        'telegram_id': row[1],
                        'username': row[2],
                        'first_name': row[3]
                    },
                    'started_at': row[4],
                    'ended_at': row[5],
                    'total_messages': row[6],
                    'duration_seconds': row[7],
                    'messages': messages
                })
            
            return conversations
    
    async def cleanup_old_data(self, keep_days: int = 90):
        """Удаляет старые данные, оставляя только последние keep_days дней"""
        async with aiosqlite.connect(self.db_path) as db:
            # Удаляем старые сообщения
            cursor = await db.execute("""
                DELETE FROM messages 
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (keep_days,))
            deleted_messages = cursor.rowcount
            
            # Удаляем старые диалоги
            cursor = await db.execute("""
                DELETE FROM conversations 
                WHERE started_at < datetime('now', '-' || ? || ' days')
            """, (keep_days,))
            deleted_conversations = cursor.rowcount
            
            await db.commit()
            
            logger.info(f"Очистка БД: удалено {deleted_messages} сообщений и {deleted_conversations} диалогов старше {keep_days} дней")
            
            return {
                'deleted_messages': deleted_messages,
                'deleted_conversations': deleted_conversations
            } 