from telegram.ext import ConversationHandler

# Состояния бота
START, INTRO_PART1, INTRO_PART2, GHOST_SELECTION, IN_GHOST, FINAL_PART1, FINAL_PART2, ENDING_PART1, ENDING_PART2 = range(9)

# Ключи для хранения данных пользователя
USER_GHOSTS_ORDER = 'user_ghosts_order'
USER_PASSED_GHOSTS = 'user_passed_ghosts'
USER_CURRENT_GHOST = 'user_current_ghost'
USER_COLLECTED_RUNES = 'user_collected_runes'
USER_FINAL_PASSED = 'user_final_passed'
USER_GHOST_RUNE_MAPPING = 'user_ghost_rune_mapping'

# Новые ключи для управления сообщениями
USER_PREVIOUS_USER_MESSAGE_ID = 'user_previous_user_message_id'
USER_PREVIOUS_BOT_MESSAGE_ID = 'user_previous_bot_message_id'