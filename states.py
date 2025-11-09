from telegram.ext import ConversationHandler

# Состояния бота
START, INTRO_PART1, INTRO_PART2, LEVEL_INTRO, AWAITING_INPUT, LEVEL_COMPLETE, VICTORY = range(7)

# Ключи для хранения данных пользователя
USER_LEVEL = 'user_level'
USER_HISTORY = 'user_history'