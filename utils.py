import logging
from typing import Dict, Any

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