from typing import Dict, Any
from config import Config

class StoryManager:
    """Управление историями и текстами для каждого призрака"""
    
    @staticmethod
    def get_intro_part1() -> str:
        return (
            "Загрузка интерфейса «SpiritLink v.1.0»...\n"
            "Установка связи... Успешно.\n\n"
            "Ты очнулся. Перед глазами — мерцающий терминал. Сквозь шум помех проступает голос наставника, старого мастера Элиаса.\n\n"
            "«Проснись, неофит. Мир, который видят обычные люди — лишь тонкая пленка. Под ней — Царство Эха, место, где застряли потерянные души. Они питаются нашей энергией, порождают хаос. Твоя задача — находить их и освобождать, узнавая Слово-Якорь. Это то, что держит их в нашем мире: обида, клятва, память... Узнай Слово — введи его в SpiritLink, и дух обретет покой.\n\n"
            "Каждый дух сильнее предыдущего. Они научились скрывать свои якоря. Будь хитер, задавай правильные вопросы. Используй «SpiritLink». И помни: некоторые духи... ждали тебя особо.»"
        )
    
    @staticmethod
    def get_intro_part2() -> str:
        return (
            "Голос наставника, мастера Элиаса, обрывается.\n\n"
            "Перед тобой открывается список сигналов. Пять потерянных душ. Ты можешь выбрать любую."
        )
    
    @staticmethod
    def get_ghost_intro(ghost_id: int) -> str:
        """Получить первое сообщение для призрака"""
        ghost = Config.GHOSTS[ghost_id]
        return (
            f">>> ВЫБРАН: {ghost['location']}\n\n"
            f"Природа: {ghost['nature']}.\n"
            f"Идентификация: {ghost['identification']}.\n\n"
            f"{ghost['description']}"
        )
    
    @staticmethod
    def get_ghost_completion(ghost_id: int, rune_index: int) -> str:
        """Получить второе сообщение для призрака после освобождения"""
        ghost = Config.GHOSTS[ghost_id]
        
        if rune_index >= len(Config.RUNES):
            rune_index = len(Config.RUNES) - 1
        
        rune_data = Config.RUNES[rune_index]
        
        return (
            f"Слово-Якорь введено в «SpiritLink»: \"{ghost['password']}\"\n\n"
            f"{ghost['completion_description']}\n\n"
            f"История поглощенного призрака:\n"
            f"{ghost['story']}\n\n"
            f"Ты получаешь:\n"
            f"- {rune_data['rune']}\n"
            f"- Фрагмент памяти: {rune_data['memory']}"
        )
    
    @staticmethod
    def get_spiritlink_instruction() -> str:
        return (
            "\n\nИспользуй «SpiritLink», чтобы вступить в диалог с призраком. Задавай вопросы, ищи ключ к его боли. "
            "Когда познаешь суть его привязанности к этому миру — введи Слово-Якорь в систему. "
            "«SpiritLink» поглотит духа, а затем, прочитав отпечаток его памяти, откроет тебе историю, что сделала его пленником этого места."
        )
    
    @staticmethod
    def get_empty_location_message() -> str:
        return "Кажется здесь теперь пусто..."
    
    @staticmethod
    def get_silence_message() -> str:
        return "в ответ лишь тишина"