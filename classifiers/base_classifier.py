from abc import ABC, abstractmethod
from typing import List

class BaseClassifier(ABC):
    """Абстрактный базовый класс для всех классификаторов"""
    
    @abstractmethod
    def classify(self, text: str) -> int:
        """
        Классифицирует текст
        Возвращает: 1 - вредоносный, 0 - безопасный
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Возвращает имя классификатора"""
        pass

# Примеры реализации классификаторов (заглушки)
class Classifier1(BaseClassifier):
    def classify(self, text: str) -> int:
        return 1
    
    def get_name(self) -> str:
        return "classifier1"

class Classifier2(BaseClassifier):
    def classify(self, text: str) -> int:
        return 1
    
    def get_name(self) -> str:
        return "classifier2"

class Classifier3(BaseClassifier):
    def classify(self, text: str) -> int:
        return 1
    
    def get_name(self) -> str:
        return "classifier3"