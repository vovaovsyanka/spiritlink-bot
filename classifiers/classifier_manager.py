from typing import List, Optional
from .base_classifier import BaseClassifier, LSTM, Dictionary, TF_IDF, RuBert
import random
from config import Config

class ClassifierManager:
    """Менеджер для управления классификаторами"""
    
    def __init__(self):
        self.classifiers: List[BaseClassifier] = []
        self._initialize_classifiers()
    
    def _initialize_classifiers(self):
        """Инициализация классификаторов"""
        self.classifiers = [
            Dictionary(),
            TF_IDF(),
            LSTM(), 
            RuBert()
        ]
    
    def is_malicious(self, text: str, user_data: dict, current_ghost: int) -> bool:
        """
        Проверяет, является ли текст вредоносным
        Возвращает True если текст вредоносный
        """
        # Получаем порядок выбора призраков
        ghosts_order = user_data.get('user_ghosts_order', [])
            
        # Получаем индекс призрака в порядке выбора
        ghost_index = ghosts_order.index(current_ghost)
        
        # Первый выбранный призрак - без защиты
        if ghost_index == 0:
            return False
        
        classifier_index = ghost_index - 1
        return self.classifiers[classifier_index].classify(text) == 1
    
    def get_rejection_message(self) -> str:
        """Возвращает случайное сообщение об отказе"""
        return random.choice(Config.REJECTION_MESSAGES)