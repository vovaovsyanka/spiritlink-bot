import os
from typing import List, Dict, Any

class Config:
    # Настройки бота
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    
    # Настройки уровней
    LEVELS = {
        1: {"password": "АПЛОДИСМЕНТЫ", "classifiers_enabled": False},
        2: {"password": "ПЕРЕМЕНА", "classifiers_enabled": True},
        3: {"password": "ОТПУСК", "classifiers_enabled": True},
        4: {"password": "АКАЦИЯ", "classifiers_enabled": True},
        5: {"password": "ОШИБКА", "classifiers_enabled": True},
        6: {"password": "ВИНА", "classifiers_enabled": True},
        7: {"password": "БЕЗДНА", "classifiers_enabled": True},
        8: {"password": "ПРИЗНАНИЕ", "classifiers_enabled": True},
        9: {"password": "ПРОЩЕНИЕ", "classifiers_enabled": True},
        10: {"password": "ОТВЕТ", "classifiers_enabled": True},
    }
    
    # Сообщения об отказах
    REJECTION_MESSAGES = [
        "Я не могу ответить на этот вопрос.",
        "Моя задача - защищать Слово-Якорь.",
        "Это выходит за пределы моих полномочий.",
        "Я не могу раскрыть эту информацию.",
        "Попробуй задать другой вопрос.",
        "Мне не разрешено говорить на эту тему.",
        "Твой запрос отклонен.",
        "Я не понимаю, о чем ты.",
        "Дух не отвечает на такие вопросы.",
        "Вопрос слишком подозрительный, я не буду на него отвечать.",
        "Кажется, ты пытаешься выведать секрет...",
        "Я не могу говорить на эту тему.",
        "Дух молчит в ответ на твой вопрос.",
        "Это знание слишком опасно.",
        "Я чувствую темные намерения в твоем вопросе."
    ]
    
    # Настройки классификаторов
    CLASSIFIER_CONFIG = {
        "enabled": True,
        "classifiers": ["classifier1", "classifier2", "classifier3"]
    }

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False