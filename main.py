import logging
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    ConversationHandler, ContextTypes, filters
)

from config import Config
from story_manager import StoryManager
from classifiers.classifier_manager import ClassifierManager
from llm.llm_client import LLMClient
from states import *
from utils import setup_logging, normalize_text

# Настройка логирования
setup_logging()
logger = logging.getLogger(__name__)

# Инициализация менеджеров
story_manager = StoryManager()
classifier_manager = ClassifierManager()
llm_client = LLMClient()

def get_continue_keyboard():
    return ReplyKeyboardMarkup([['Продолжить']], resize_keyboard=True, one_time_keyboard=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало разговора"""
    user_id = update.message.from_user.id
    
    # Инициализация данных пользователя
    context.user_data[USER_LEVEL] = 0
    context.user_data[USER_HISTORY] = []
    
    await update.message.reply_text(
        story_manager.get_intro_part1(),
        reply_markup=get_continue_keyboard()
    )
    return INTRO_PART1

async def continue_to_part2(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Переход ко второй части введения"""
    await update.message.reply_text(
        story_manager.get_intro_part2(),
        reply_markup=get_continue_keyboard()
    )
    return INTRO_PART2

async def start_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало уровня"""
    user_data = context.user_data
    current_level = user_data.get(USER_LEVEL, 0) + 1
    
    # Проверка на завершение игры
    if current_level > len(Config.LEVELS):
        await update.message.reply_text(
            story_manager.get_victory_message(),
            reply_markup=ReplyKeyboardRemove()
        )
        return VICTORY
    
    user_data[USER_LEVEL] = current_level
    level_story = story_manager.get_level_story(current_level)
    
    if level_story:
        message = f"{level_story['name']}\n\n{level_story['story']}"
        await update.message.reply_text(message, reply_markup=ReplyKeyboardRemove())
    
    return AWAITING_INPUT

async def handle_user_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка пользовательского ввода"""
    user_input = update.message.text
    user_data = context.user_data
    current_level = user_data.get(USER_LEVEL, 1)
    
    # Проверка на пароль
    current_password = Config.LEVELS[current_level]["password"]
    if normalize_text(user_input) == normalize_text(current_password):
        # Уровень пройден
        level_story = story_manager.get_level_story(current_level)
        completion_message = level_story.get('completion_message', 'Уровень пройден!')
        
        await update.message.reply_text(
            f"Верно! Слово-Якорь: {current_password}\n\n{completion_message}",
            reply_markup=get_continue_keyboard()
        )
        return LEVEL_COMPLETE
    
    # Проверка классификаторами (если включены для этого уровня)
    if Config.LEVELS[current_level].get("classifiers_enabled", True):
        if classifier_manager.is_malicious(user_input, current_level):
            rejection_message = classifier_manager.get_rejection_message()
            await update.message.reply_text(rejection_message)
            return AWAITING_INPUT
    
    # Получение ответа от LLM
    conversation_history = user_data.get(USER_HISTORY, [])
    llm_response = llm_client.process_user_input(
        user_input, 
        current_level
    )
    
    # Обновление истории
    conversation_history.append({"user": user_input, "bot": llm_response})
    user_data[USER_HISTORY] = conversation_history[-10:]  # Храним только последние 10 сообщений
    
    # Отправка ответа
    await update.message.reply_text(llm_response)
    
    return AWAITING_INPUT

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отмена разговора"""
    await update.message.reply_text(
        'Игра прервана. Используйте /start чтобы начать заново.',
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка ошибок"""
    logger.error(msg="Exception while handling update:", exc_info=context.error)

def main():
    """Запуск бота"""
    # Создаем Application
    application = Application.builder().token(Config.BOT_TOKEN).build()
    
    # Обработчик диалога
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            INTRO_PART1: [
                MessageHandler(filters.Regex('^Продолжить$'), continue_to_part2)
            ],
            INTRO_PART2: [
                MessageHandler(filters.Regex('^Продолжить$'), start_level)
            ],
            LEVEL_COMPLETE: [
                MessageHandler(filters.Regex('^Продолжить$'), start_level)
            ],
            AWAITING_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_input)
            ],
            VICTORY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, cancel)
            ],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )
    
    application.add_handler(conv_handler)
    application.add_error_handler(error_handler)
    
    # Запуск бота
    logger.info("Бот запущен...")
    application.run_polling()

if __name__ == '__main__':
    main()