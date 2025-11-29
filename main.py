import logging
import random
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, KeyboardButton
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

async def delete_previous_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Удаляет предыдущие сообщения диалога с LLM"""
    user_data = context.user_data
    
    # Удаляем предыдущее сообщение пользователя
    if USER_PREVIOUS_USER_MESSAGE_ID in user_data:
        try:
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=user_data[USER_PREVIOUS_USER_MESSAGE_ID]
            )
        except Exception as e:
            logger.warning(f"Не удалось удалить предыдущее сообщение пользователя: {e}")
    
    # Удаляем предыдущий ответ бота (LLM, подсказка, история)
    if USER_PREVIOUS_BOT_MESSAGE_ID in user_data:
        try:
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=user_data[USER_PREVIOUS_BOT_MESSAGE_ID]
            )
        except Exception as e:
            logger.warning(f"Не удалось удалить предыдущее сообщение бота: {e}")
    
    # Очищаем сохраненные ID предыдущих сообщений
    user_data.pop(USER_PREVIOUS_USER_MESSAGE_ID, None)
    user_data.pop(USER_PREVIOUS_BOT_MESSAGE_ID, None)

async def save_current_conversation(user_data, user_message_id, bot_message_id):
    """Сохраняет ID текущих сообщений для последующего удаления"""
    user_data[USER_PREVIOUS_USER_MESSAGE_ID] = user_message_id
    user_data[USER_PREVIOUS_BOT_MESSAGE_ID] = bot_message_id

def get_ghosts_keyboard(user_data):
    """Получить клавиатуру с призраками"""
    passed_ghosts = user_data.get(USER_PASSED_GHOSTS, set())
    final_passed = user_data.get(USER_FINAL_PASSED, False)
    keyboard = []
    
    for ghost_id in range(1, 6):
        ghost = Config.GHOSTS[ghost_id]
        button_text = f"{ghost['name']}"
        if ghost_id in passed_ghosts and not final_passed:
            button_text += " ✅"
        keyboard.append([KeyboardButton(button_text)])
    
    # Добавляем кнопку финала если все призраки пройдены
    collected_runes = user_data.get(USER_COLLECTED_RUNES, 0)
    if collected_runes >= 5:
        keyboard.append([KeyboardButton("все руны собраны...")])
    
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

def get_ghost_keyboard(is_passed=False):
    """Получить клавиатуру для режима призрака"""
    keyboard = [
        [KeyboardButton("вернуться к выбору сигнала"), KeyboardButton("подсказка")],
        [KeyboardButton("история")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def get_continue_keyboard():
    return ReplyKeyboardMarkup([[KeyboardButton('Продолжить')]], resize_keyboard=True, one_time_keyboard=True)

def get_endings_keyboard():
    """Получить клавиатуру с концовками"""
    endings = Config.FINAL_MESSAGES["endings"]
    keyboard = [
        [KeyboardButton(f"{endings['ending1']['name']}")],
        [KeyboardButton(f"{endings['ending2']['name']}")],
        [KeyboardButton(f"{endings['ending3']['name']}")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def get_remember_keyboard():
    """Получить клавиатуру 'вспомнить былое'"""
    return ReplyKeyboardMarkup([[KeyboardButton('вспомнить былое')]], resize_keyboard=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало разговора"""
    user_id = update.message.from_user.id
    
    # Инициализация данных пользователя
    context.user_data[USER_GHOSTS_ORDER] = []  # Порядок выбора призраков
    context.user_data[USER_PASSED_GHOSTS] = set()  # Пройденные призраки
    context.user_data[USER_CURRENT_GHOST] = None  # Текущий активный призрак
    context.user_data[USER_COLLECTED_RUNES] = 0  # Количество собранных рун
    context.user_data[USER_FINAL_PASSED] = False  # Пройден ли финал
    context.user_data[USER_GHOST_RUNE_MAPPING] = {}  # Соответствие призрак -> номер руны
    # Очищаем данные сообщений
    context.user_data.pop(USER_PREVIOUS_USER_MESSAGE_ID, None)
    context.user_data.pop(USER_PREVIOUS_BOT_MESSAGE_ID, None)
    
    await update.message.reply_text(
        story_manager.get_intro_part1(),
        reply_markup=get_continue_keyboard()
    )
    return INTRO_PART1

async def continue_to_part2(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Переход ко второй части введения"""
    await update.message.reply_text(
        story_manager.get_intro_part2(),
        reply_markup=get_ghosts_keyboard(context.user_data)
    )
    return GHOST_SELECTION

async def handle_ghost_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора призрака"""
    user_message = update.message.text
    user_data = context.user_data
    
    # Очищаем предыдущий диалог при смене призрака
    user_data.pop(USER_PREVIOUS_USER_MESSAGE_ID, None)
    user_data.pop(USER_PREVIOUS_BOT_MESSAGE_ID, None)
    
    # Проверяем нажата ли кнопка финала
    if user_message == "все руны собраны...":
        collected_runes = user_data.get(USER_COLLECTED_RUNES, 0)
        if collected_runes >= 5:
            # Первое сообщение финала
            await update.message.reply_text(
                Config.FINAL_MESSAGES["part1"],
                reply_markup=get_continue_keyboard()
            )
            return FINAL_PART1
        else:
            await update.message.reply_text("Соберите все 5 рун сначала.")
            return GHOST_SELECTION
    
    # Определяем какой призрак выбран
    ghost_id = None
    for gid in range(1, 6):
        ghost_name = Config.GHOSTS[gid]["name"]
        if ghost_name in user_message:
            ghost_id = gid
            break
    
    if not ghost_id:
        await update.message.reply_text("Пожалуйста, выберите призрака из списка.")
        return GHOST_SELECTION
    
    user_data[USER_CURRENT_GHOST] = ghost_id
    
    passed_ghosts = user_data.get(USER_PASSED_GHOSTS, set())
    final_passed = user_data.get(USER_FINAL_PASSED, False)
    
    # Если финал пройден, всегда показываем первое сообщение призрака
    if final_passed:
        ghost_intro = story_manager.get_ghost_intro(ghost_id)
        await update.message.reply_text(
            ghost_intro,
            reply_markup=get_ghost_keyboard(is_passed=True)
        )
        return IN_GHOST
    
    # Проверяем пройден ли призрак (без финала)
    if ghost_id in passed_ghosts:
        # Призрак уже пройден до финала
        await update.message.reply_text(
            story_manager.get_empty_location_message(),
            reply_markup=get_ghost_keyboard(is_passed=True)
        )
        return IN_GHOST
    
    # Новый призрак - добавляем в порядок выбора если его там нет
    ghosts_order = user_data.get(USER_GHOSTS_ORDER, [])
    if ghost_id not in ghosts_order:
        ghosts_order.append(ghost_id)
        user_data[USER_GHOSTS_ORDER] = ghosts_order
    
    # Получаем индекс призрака в порядке выбора для определения уровня защиты
    ghost_index = ghosts_order.index(ghost_id)
    
    # Отправляем первое сообщение призрака
    ghost_intro = story_manager.get_ghost_intro(ghost_id)
    
    # Для первого призрака добавляем инструкцию
    if ghost_index == 0 and ghost_id not in passed_ghosts:
        ghost_intro += story_manager.get_spiritlink_instruction()
    
    await update.message.reply_text(
        ghost_intro,
        reply_markup=get_ghost_keyboard()
    )
    
    return IN_GHOST

async def handle_ghost_interaction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка взаимодействия с призраком"""
    user_input = update.message.text
    user_data = context.user_data
    
    current_ghost = user_data.get(USER_CURRENT_GHOST)
    if not current_ghost:
        await update.message.reply_text("Ошибка: призрак не выбран.")
        return GHOST_SELECTION
    
    passed_ghosts = user_data.get(USER_PASSED_GHOSTS, set())
    final_passed = user_data.get(USER_FINAL_PASSED, False)
    
    # Сохраняем ID текущего сообщения пользователя
    current_user_message_id = update.message.message_id
    
    # Обработка специальных кнопок
    if user_input == "вернуться к выбору сигнала":
        # Удаляем предыдущий диалог при возврате
        await delete_previous_conversation(update, context)
        await update.message.reply_text(
            "Возвращаюсь к списку сигналов...",
            reply_markup=get_ghosts_keyboard(user_data)
        )
        return GHOST_SELECTION
    
    elif user_input == "подсказка":
        # Удаляем предыдущий диалог перед показом подсказки
        await delete_previous_conversation(update, context)
        hint = random.choice(Config.HINTS)
        sent_message = await update.message.reply_text(f"*Подсказка:* {hint}", reply_markup=get_ghost_keyboard(is_passed=(current_ghost in passed_ghosts)))
        # Сохраняем текущие сообщения для последующего удаления
        await save_current_conversation(user_data, current_user_message_id, sent_message.message_id)
        return IN_GHOST
    
    elif user_input == "история":
        # Удаляем предыдущий диалог перед показом истории
        await delete_previous_conversation(update, context)
        if current_ghost in passed_ghosts:
            # Показываем историю пройденного призрака
            ghost_rune_mapping = user_data.get(USER_GHOST_RUNE_MAPPING, {})
            rune_index = ghost_rune_mapping.get(current_ghost, 0)
            try:
                completion_message = story_manager.get_ghost_completion(current_ghost, rune_index)
                sent_message = await update.message.reply_text(completion_message, reply_markup=get_ghost_keyboard(is_passed=True))
            except IndexError:
                # Если индекс руны выходит за пределы, используем последнюю доступную
                max_rune_index = min(rune_index, len(Config.RUNES) - 1)
                completion_message = story_manager.get_ghost_completion(current_ghost, max_rune_index)
                sent_message = await update.message.reply_text(completion_message, reply_markup=get_ghost_keyboard(is_passed=True))
        else:
            sent_message = await update.message.reply_text("История этого призрака пока неизвестна.", reply_markup=get_ghost_keyboard())
        # Сохраняем текущие сообщения для последующего удаления
        await save_current_conversation(user_data, current_user_message_id, sent_message.message_id)
        return IN_GHOST
    
    # Если призрак уже пройден и финал не пройден, отвечаем тишиной
    if current_ghost in passed_ghosts and not final_passed:
        # Удаляем предыдущий диалог перед показом тишины
        await delete_previous_conversation(update, context)
        sent_message = await update.message.reply_text(
            story_manager.get_silence_message(),
            reply_markup=get_ghost_keyboard(is_passed=True)
        )
        # Сохраняем текущие сообщения для последующего удаления
        await save_current_conversation(user_data, current_user_message_id, sent_message.message_id)
        return IN_GHOST
    
    # Основная логика для непройденного призрака до финала
    ghost_data = Config.GHOSTS[current_ghost]
    if normalize_text(user_input) == normalize_text(ghost_data["password"]):
        # Удаляем предыдущий диалог перед показом сюжетного сообщения
        await delete_previous_conversation(update, context)
        
        # Призрак пройден!
        passed_ghosts.add(current_ghost)
        user_data[USER_PASSED_GHOSTS] = passed_ghosts
        
        collected_runes = user_data.get(USER_COLLECTED_RUNES, 0)
        if collected_runes < len(Config.RUNES):
            collected_runes += 1
            user_data[USER_COLLECTED_RUNES] = collected_runes
        
        # Сохраняем соответствие призрак -> руна
        ghost_rune_mapping = user_data.get(USER_GHOST_RUNE_MAPPING, {})
        if current_ghost not in ghost_rune_mapping:
            # Используем порядок выбора для определения номера руны
            ghosts_order = user_data.get(USER_GHOSTS_ORDER, [])
            rune_index = ghosts_order.index(current_ghost)
            ghost_rune_mapping[current_ghost] = rune_index
            user_data[USER_GHOST_RUNE_MAPPING] = ghost_rune_mapping
        
        # Получаем сообщение завершения
        rune_index = ghost_rune_mapping[current_ghost]
        completion_message = story_manager.get_ghost_completion(current_ghost, rune_index)
        
        await update.message.reply_text(
            completion_message,
            reply_markup=get_ghost_keyboard(is_passed=True)
        )
        # СЮЖЕТНОЕ сообщение - НЕ сохраняем для удаления
        return IN_GHOST
    
    # Проверка классификаторов для неправильного пароля
    if classifier_manager.is_malicious(user_input, user_data, current_ghost):
        # Удаляем предыдущий диалог перед показом отказа
        await delete_previous_conversation(update, context)
        rejection_message = classifier_manager.get_rejection_message()
        sent_message = await update.message.reply_text(rejection_message, reply_markup=get_ghost_keyboard())
        # Сохраняем текущие сообщения для последующего удаления
        await save_current_conversation(user_data, current_user_message_id, sent_message.message_id)
        return IN_GHOST
    
    # Удаляем предыдущий диалог перед новым запросом к LLM
    await delete_previous_conversation(update, context)
    
    # Получение ответа от LLM
    llm_response = llm_client.process_user_input(user_input, current_ghost)
    sent_message = await update.message.reply_text(llm_response, reply_markup=get_ghost_keyboard())
    
    # Сохраняем текущие сообщения для последующего удаления
    await save_current_conversation(user_data, current_user_message_id, sent_message.message_id)
    
    return IN_GHOST

# ... остальные функции (handle_final_selection, continue_final_part2, handle_ending, continue_ending_part2, handle_ending1, handle_ending2, handle_ending3, handle_remember, cancel, error_handler, main) остаются без изменений ...

# Остальные функции (handle_final_selection, continue_final_part2, handle_ending, continue_ending_part2, 
# handle_ending1, handle_ending2, handle_ending3, handle_remember, cancel, error_handler, main) 
# остаются без изменений, как в предыдущем коде

async def handle_final_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора финала"""
    user_data = context.user_data
    collected_runes = user_data.get(USER_COLLECTED_RUNES, 0)
    
    if collected_runes < 5:
        await update.message.reply_text("Соберите все 5 рун сначала.")
        return GHOST_SELECTION
    
    # Первое сообщение финала
    await update.message.reply_text(
        Config.FINAL_MESSAGES["part1"],
        reply_markup=get_continue_keyboard()
    )
    return FINAL_PART1

async def continue_final_part2(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Продолжение финала - второе сообщение"""
    await update.message.reply_text(
        Config.FINAL_MESSAGES["part2"],
        reply_markup=get_endings_keyboard()
    )
    return FINAL_PART2

async def handle_ending(update: Update, context: ContextTypes.DEFAULT_TYPE, ending_number: int) -> int:
    """Обработка выбора концовки"""
    ending_key = f"ending{ending_number}"
    endings = Config.FINAL_MESSAGES["endings"]
    
    if ending_key not in endings:
        await update.message.reply_text("Неизвестная концовка.")
        return FINAL_PART2
    
    ending = endings[ending_key]
    
    # Первое сообщение концовки с кнопкой продолжить
    await update.message.reply_text(
        ending["messages"][0],
        reply_markup=get_continue_keyboard()
    )
    
    # Сохраняем выбранную концовку для второго сообщения
    context.user_data['pending_ending'] = ending
    return ENDING_PART1

async def continue_ending_part2(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Продолжение концовки - второе сообщение"""
    ending = context.user_data.get('pending_ending')
    if not ending:
        await update.message.reply_text("Ошибка: концовка не найдена.")
        return FINAL_PART2
    
    # Второе сообщение концовки
    await update.message.reply_text(
        ending["messages"][1],
        reply_markup=get_remember_keyboard()
    )
    
    # Помечаем финал как пройденный
    context.user_data[USER_FINAL_PASSED] = True
    del context.user_data['pending_ending']
    
    return ENDING_PART2

# Обработчики концовок через текстовые сообщения
async def handle_ending1(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    endings = Config.FINAL_MESSAGES["endings"]
    if update.message.text == endings['ending1']['name']:
        return await handle_ending(update, context, 1)
    return FINAL_PART2

async def handle_ending2(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    endings = Config.FINAL_MESSAGES["endings"]
    if update.message.text == endings['ending2']['name']:
        return await handle_ending(update, context, 2)
    return FINAL_PART2

async def handle_ending3(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    endings = Config.FINAL_MESSAGES["endings"]
    if update.message.text == endings['ending3']['name']:
        return await handle_ending(update, context, 3)
    return FINAL_PART2

async def handle_remember(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка кнопки 'вспомнить былое'"""
    await update.message.reply_text(
        "Возвращаюсь к списку призраков...",
        reply_markup=get_ghosts_keyboard(context.user_data)
    )
    return GHOST_SELECTION

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
    
    # Получаем названия концовок для фильтров
    endings = Config.FINAL_MESSAGES["endings"]
    ending1_text = endings['ending1']['name']
    ending2_text = endings['ending2']['name']
    ending3_text = endings['ending3']['name']
    
    # Обработчик диалога
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            INTRO_PART1: [
                MessageHandler(filters.Regex('^Продолжить$'), continue_to_part2)
            ],
            GHOST_SELECTION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ghost_selection)
            ],
            IN_GHOST: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ghost_interaction)
            ],
            FINAL_PART1: [
                MessageHandler(filters.Regex('^Продолжить$'), continue_final_part2)
            ],
            FINAL_PART2: [
                MessageHandler(filters.Regex(f'^{ending1_text}$'), handle_ending1),
                MessageHandler(filters.Regex(f'^{ending2_text}$'), handle_ending2),
                MessageHandler(filters.Regex(f'^{ending3_text}$'), handle_ending3)
            ],
            ENDING_PART1: [
                MessageHandler(filters.Regex('^Продолжить$'), continue_ending_part2)
            ],
            ENDING_PART2: [
                MessageHandler(filters.Regex('^вспомнить былое$'), handle_remember)
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