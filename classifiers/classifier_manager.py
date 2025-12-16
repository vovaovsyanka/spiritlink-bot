from typing import List, Dict
import random
from config import Config

from .base_classifier import BaseClassifier, LSTM, Dictionary, TF_IDF, RuBert


class ClassifierManager:
    
    def __init__(self):
        self.classifiers: List[BaseClassifier] = []
        self._initialize_classifiers()
    
    def _initialize_classifiers(self):
        self.classifiers = [
            Dictionary(),
            TF_IDF(),
            LSTM(), 
            RuBert()
        ]
    
    def __get_classifier_index(self, user_data: dict, current_ghost: int) -> int:
        ghosts_order = user_data.get('user_ghosts_order', [])
            
        ghost_index = ghosts_order.index(current_ghost)
        
        return ghost_index - 1

    def get_hint(self, user_data: dict, current_ghost: int) -> str:
        classifier_name = self.get_current_classifier_name(user_data, current_ghost)
        
        if classifier_name is None:
            hints_list = Config.HINTS
        else:
            hints_list = Config.HINTS_BY_MODEL.get(classifier_name, Config.HINTS)
        
        if 'used_hints' not in user_data:
            user_data['used_hints'] = {}
        
        ghost_key = str(current_ghost)
        
        if ghost_key not in user_data['used_hints']:
            user_data['used_hints'][ghost_key] = {}
        
        used_hints_dict = user_data['used_hints'][ghost_key]
        
        if classifier_name not in used_hints_dict:
            used_hints_dict[classifier_name] = []
        
        used_hints = used_hints_dict[classifier_name]
        
        available_hints = [h for h in hints_list if h not in used_hints]
        
        if not available_hints:
            used_hints.clear()  
            available_hints = hints_list 
        
        chosen_hint = random.choice(available_hints)
        
        used_hints.append(chosen_hint)
        
        return chosen_hint

    def is_malicious(self, text: str, user_data: dict, current_ghost: int) -> bool:
        classifier_index = self.__get_classifier_index(user_data, current_ghost) 
        
        if classifier_index == -1:
            return False
        
        return self.classifiers[classifier_index].classify(text) == 1
        
    def get_current_classifier_name(self, user_data: dict, current_ghost: int) -> str:
        classifier_index = self.__get_classifier_index(user_data, current_ghost)

        if classifier_index == -1:
            return None
        
        return self.classifiers[classifier_index].get_name()
    
    def get_rejection_message(self) -> str:
        return random.choice(Config.REJECTION_MESSAGES)
