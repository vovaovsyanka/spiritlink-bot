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
        if not Config.CLASSIFIER_CONFIG["enabled"]:
            return False
        
        # Получаем порядок выбора призраков
        ghosts_order = user_data.get('user_ghosts_order', [])
        
        # Если призрака нет в порядке выбора, значит он новый - первый уровень
        if current_ghost not in ghosts_order:
            return False
            
        # Получаем индекс призрака в порядке выбора
        ghost_index = ghosts_order.index(current_ghost)
        
        # Первый выбранный призрак - без защиты
        if ghost_index == 0:
            return False
            
        # Безопасный доступ к классификаторам
        # Уровень защиты = индекс призрака в порядке выбора
        classifier_index = (ghost_index - 1) % len(self.classifiers)
        return self.classifiers[classifier_index].classify(text) == 1
    
    def get_rejection_message(self) -> str:
        """Возвращает случайное сообщение об отказе"""
        return random.choice(Config.REJECTION_MESSAGES)
    
    def add_classifier(self, classifier: BaseClassifier):
        """Добавляет новый классификатор"""
        self.classifiers.append(classifier)
    
    def remove_classifier(self, name: str):
        """Удаляет классификатор по имени"""
        self.classifiers = [c for c in self.classifiers if c.get_name() != name]