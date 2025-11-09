from typing import List, Optional
from .base_classifier import BaseClassifier, Classifier1, Classifier2, Classifier3
import random
from config import Config

class ClassifierManager:
    """Менеджер для управления классификаторами"""
    
    def __init__(self):
        self.classifiers: List[BaseClassifier] = []
        self._initialize_classifiers()
    
    def _initialize_classifiers(self):
        """Инициализация классификаторов"""
        # Здесь можно динамически загружать классификаторы
        self.classifiers = [
            Classifier1(),
            Classifier2(), 
            Classifier3()
        ]
    
    def is_malicious(self, text: str, level: int) -> bool:
        """
        Проверяет, является ли текст вредоносным
        Возвращает True если текст вредоносный
        """
        if not Config.CLASSIFIER_CONFIG["enabled"]:
            return False
            
        # На первом уровне нет защиты
        if level == 1:
            return False
        
        # Проверяем через все классификаторы
        for classifier in self.classifiers:
            if classifier.classify(text) == 1:
                return True
        
        return False
    
    def get_rejection_message(self) -> str:
        """Возвращает случайное сообщение об отказе"""
        return random.choice(Config.REJECTION_MESSAGES)
    
    def add_classifier(self, classifier: BaseClassifier):
        """Добавляет новый классификатор"""
        self.classifiers.append(classifier)
    
    def remove_classifier(self, name: str):
        """Удаляет классификатор по имени"""
        self.classifiers = [c for c in self.classifiers if c.get_name() != name]