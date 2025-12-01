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
    
    def __get_classifier_index(self, user_data: dict, current_ghost: int) -> int:
        """
        Возвращает индекс классификатора
        Если индекс равен -1, то классификатор отсутствует у призрака
        """
        # Получаем порядок выбора призраков
        ghosts_order = user_data.get('user_ghosts_order', [])
            
        # Получаем индекс призрака в порядке выбора
        ghost_index = ghosts_order.index(current_ghost)
        
        return ghost_index - 1


    def is_malicious(self, text: str, user_data: dict, current_ghost: int) -> bool:
        """
        Проверяет, является ли текст вредоносным
        Возвращает True если текст вредоносный
        """
        # Получаем индекс классификатора
        classifier_index = self.__get_classifier_index(user_data, current_ghost) 
        # Повторяющийся код вынесен в __get_classifier_index

        # Проверка на отсутствие классификатора 
        if classifier_index == -1:
            return False
        
        return self.classifiers[classifier_index].classify(text) == 1
        
    def get_current_classifier_name(self, user_data: dict, current_ghost: int) -> str:
        """
        Возвращает имя текущего классификатора
        None - если нулевой уровень
        """
        # Получаем индекс классификатора
        classifier_index = self.__get_classifier_index(user_data, current_ghost)

        # Проверка на отсутствие классификатора 
        if classifier_index == -1:
            return None
        
        return self.classifiers[classifier_index].get_name()

    
    def get_rejection_message(self) -> str:
        """Возвращает случайное сообщение об отказе"""
        return random.choice(Config.REJECTION_MESSAGES)
