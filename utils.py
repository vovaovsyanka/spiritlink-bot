import logging
from typing import Dict, Any
import asyncio

def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def clean_user_data(user_data: Dict[str, Any]):
    """Очистка данных пользователя"""
    keys_to_keep = [USER_LEVEL, USER_HISTORY]  # из states.py
    keys_to_remove = [key for key in user_data.keys() if key not in keys_to_keep]
    
    for key in keys_to_remove:
        del user_data[key]

def normalize_text(text: str) -> str:
    """Нормализация текста для сравнения"""
    return text.upper().strip()

async def typing_while_waiting(context, chat_id, task):
    """Показывает 'печатает…' пока task не завершится"""
    while not task.done():
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        await asyncio.sleep(4)  # Telegram сбрасывает typing примерно раз в 5 секунд

def ghost_to_image(ghost_name: str):
    """Возвращает название картинки соответствующее призраку"""
    if ghost_name == "Дух Заброшенного Театра":
        return "Пьеро.jpeg"
    elif ghost_name == "Дух Пыльной Библиотеки":
        return "Лиза.jpeg"
    elif ghost_name == "Дух Заводского Цеха":
        return "Борис.jpeg"
    elif ghost_name == "Дух Заброшенной Оранжереи":
        return "Флора.jpeg"
    elif ghost_name == "Дух Старого Серверного Зала":
        return "Цифра.jpeg"
    return None
