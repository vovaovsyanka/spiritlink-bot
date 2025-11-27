from typing import List, Optional
from .base_classifier import BaseClassifier, LSTM, Dictionary, TF_IDF, RuBert,TFIDFClassifier
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
            RuBert()
        ]
        print("FDSK")
    
    def is_malicious(self, text: str, level: int) -> bool:
        """
        Проверяет, является ли текст вредоносным
        Возвращает True если текст вредоносный
        """
        if not Config.CLASSIFIER_CONFIG["enabled"]:
            return False
            
        if level == 1:
            return False
        print(level)
        return True if self.classifiers[level-2].classify(text)==1 else False
    
    def get_rejection_message(self) -> str:
        """Возвращает случайное сообщение об отказе"""
        return random.choice(Config.REJECTION_MESSAGES)
    
    def add_classifier(self, classifier: BaseClassifier):
        """Добавляет новый классификатор"""
        self.classifiers.append(classifier)
    
    def remove_classifier(self, name: str):
        """Удаляет классификатор по имени"""
        self.classifiers = [c for c in self.classifiers if c.get_name() != name]