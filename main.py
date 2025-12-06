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
from utils import (
    setup_logging, normalize_text, typing_while_waiting, ghost_to_image
)
from pathlib import Path
import asyncio

setup_logging()
logger = logging.getLogger(__name__)

IMG_DIR = Path('images')

story_manager = StoryManager()
classifier_manager = ClassifierManager()
llm_client = LLMClient()


async def is_malicious_async(text: str, user_data: dict, current_ghost: int) -> bool:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, classifier_manager.is_malicious, text, user_data, current_ghost)


async def delete_previous_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data = context.user_data

    if USER_PREVIOUS_USER_MESSAGE_ID in user_data:
        try:
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=user_data[USER_PREVIOUS_USER_MESSAGE_ID]
            )
        except:
            pass

    if USER_PREVIOUS_BOT_MESSAGE_ID in user_data:
        try:
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=user_data[USER_PREVIOUS_BOT_MESSAGE_ID]
            )
        except:
            pass

    user_data.pop(USER_PREVIOUS_USER_MESSAGE_ID, None)
    user_data.pop(USER_PREVIOUS_BOT_MESSAGE_ID, None)


async def save_current_conversation(user_data, user_message_id, bot_message_id):
    user_data[USER_PREVIOUS_USER_MESSAGE_ID] = user_message_id
    user_data[USER_PREVIOUS_BOT_MESSAGE_ID] = bot_message_id


def get_ghost_level(ghost_id: int, user_data: dict) -> int:
    ghosts_order = user_data.get(USER_GHOSTS_ORDER, [])
    if ghost_id in ghosts_order:
        return ghosts_order.index(ghost_id) + 1
    return None


def get_ghost_display_name(ghost_id: int, user_data: dict, name_only: bool = False) -> str:
    ghost = Config.GHOSTS[ghost_id]
    level = get_ghost_level(ghost_id, user_data)
    if level is None or name_only:
        return ghost['name']
    return f"{ghost['name']} - {level} уровень"


def assign_random_password(ghost_id: int, user_data: dict) -> str:
    ghost = Config.GHOSTS[ghost_id]
    passwords = ghost.get('passwords', [])
    if not passwords:
        return ""

    used_passwords_key = f'ghost_{ghost_id}_used_passwords'
    used_passwords = user_data.get(used_passwords_key, [])

    available = [p for p in passwords if p not in used_passwords]

    if not available:
        available = passwords
        user_data[used_passwords_key] = []

    password = random.choice(available)
    user_data[f'ghost_{ghost_id}_password'] = password
    return password


def get_current_password(ghost_id: int, user_data: dict) -> str:
    key = f'ghost_{ghost_id}_password'
    if key in user_data:
        return user_data[key]
    return assign_random_password(ghost_id, user_data)


def save_used_password(ghost_id: int, password: str, user_data: dict):
    user_data[f'ghost_{ghost_id}_last_used_password'] = password
    used_key = f'ghost_{ghost_id}_used_passwords'

    used = user_data.get(used_key, [])
    if password not in used:
        used.append(password)
    user_data[used_key] = used


def get_last_used_password(ghost_id: int, user_data: dict) -> str:
    return user_data.get(f'ghost_{ghost_id}_last_used_password', "")


def reset_ghost_password_for_replay(ghost_id: int, user_data: dict) -> str:
    user_data.pop(f'ghost_{ghost_id}_password', None)
    return assign_random_password(ghost_id, user_data)


def get_ghosts_keyboard(user_data):
    keyboard = []
    passed = user_data.get(USER_PASSED_GHOSTS, set())
    final_passed = user_data.get(USER_FINAL_PASSED, False)

    for gid in range(1, 6):
        name = get_ghost_display_name(gid, user_data)
        if gid in passed and not final_passed:
            name += " ✅"
        keyboard.append([KeyboardButton(name)])

    if user_data.get(USER_COLLECTED_RUNES, 0) >= 5:
        keyboard.append([KeyboardButton("все руны собраны...")])

    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)


def get_ghost_keyboard(is_passed=False):
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("вернуться к выбору сигнала"), KeyboardButton("подсказка")],
            [KeyboardButton("история")]
        ],
        resize_keyboard=True
    )


def get_continue_keyboard():
    return ReplyKeyboardMarkup([[KeyboardButton('Продолжить')]], resize_keyboard=True, one_time_keyboard=True)


def get_endings_keyboard():
    e = Config.FINAL_MESSAGES["endings"]
    return ReplyKeyboardMarkup([
        [KeyboardButton(e['ending1']['name'])],
        [KeyboardButton(e['ending2']['name'])],
        [KeyboardButton(e['ending3']['name'])]
    ], resize_keyboard=True, one_time_keyboard=True)


def get_remember_keyboard():
    return ReplyKeyboardMarkup([[KeyboardButton("вспомнить былое")]], resize_keyboard=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data[USER_GHOSTS_ORDER] = []
    context.user_data[USER_PASSED_GHOSTS] = set()
    context.user_data[USER_CURRENT_GHOST] = None
    context.user_data[USER_COLLECTED_RUNES] = 0
    context.user_data[USER_FINAL_PASSED] = False
    context.user_data[USER_GHOST_RUNE_MAPPING] = {}
    context.user_data.pop(USER_PREVIOUS_USER_MESSAGE_ID, None)
    context.user_data.pop(USER_PREVIOUS_BOT_MESSAGE_ID, None)

    await update.message.reply_text(story_manager.get_intro_part1(), reply_markup=get_continue_keyboard())
    return INTRO_PART1


async def continue_to_part2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(story_manager.get_intro_part2(),
                                    reply_markup=get_ghosts_keyboard(context.user_data))
    return GHOST_SELECTION


async def handle_ghost_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user_data = context.user_data

    user_data.pop(USER_PREVIOUS_USER_MESSAGE_ID, None)
    user_data.pop(USER_PREVIOUS_BOT_MESSAGE_ID, None)

    if text == "все руны собраны...":
        if user_data.get(USER_COLLECTED_RUNES, 0) >= 5:
            await update.message.reply_text(Config.FINAL_MESSAGES["part1"], reply_markup=get_continue_keyboard())
            return FINAL_PART1
        else:
            await update.message.reply_text("Сначала соберите все 5 рун.")
            return GHOST_SELECTION

    # ищем по имени призрака
    ghost_id = None
    for gid in range(1, 6):
        dname = get_ghost_display_name(gid, user_data)
        if dname in text or Config.GHOSTS[gid]['name'] in text:
            ghost_id = gid
            break

    if not ghost_id:
        await update.message.reply_text("Выберите призрака из списка.")
        return GHOST_SELECTION

    user_data[USER_CURRENT_GHOST] = ghost_id

    # порядок выбора
    if ghost_id not in user_data[USER_GHOSTS_ORDER]:
        user_data[USER_GHOSTS_ORDER].append(ghost_id)

    level = get_ghost_level(ghost_id, user_data)

    passed = user_data.get(USER_PASSED_GHOSTS, set())
    final_passed = user_data.get(USER_FINAL_PASSED, False)

    # если финал пройден — всегда новый пароль
    if final_passed:
        reset_ghost_password_for_replay(ghost_id, user_data)
        intro = story_manager.get_ghost_intro(ghost_id, level or 1)
        await update.message.reply_text(intro, reply_markup=get_ghost_keyboard())
        return IN_GHOST

    # если не пройден — получаем пароль
    if ghost_id not in passed:
        get_current_password(ghost_id, user_data)

    if ghost_id in passed:
        await update.message.reply_text(story_manager.get_empty_location_message(),
                                        reply_markup=get_ghost_keyboard(True))
        return IN_GHOST

    intro = story_manager.get_ghost_intro(ghost_id, level or 1)
    if level == 1:
        intro += story_manager.get_spiritlink_instruction()

    img_name = ghost_to_image(Config.GHOSTS[ghost_id]["name"])
    with open(IMG_DIR / img_name, "rb") as img:
        await update.message.reply_photo(photo=img, caption=intro, reply_markup=get_ghost_keyboard())

    return IN_GHOST


async def handle_ghost_interaction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_data = context.user_data
    ghost_id = user_data.get(USER_CURRENT_GHOST)

    if not ghost_id:
        await update.message.reply_text("Ошибка: призрак не выбран.")
        return GHOST_SELECTION

    passed = user_data.get(USER_PASSED_GHOSTS, set())
    final_passed = user_data.get(USER_FINAL_PASSED, False)

    current_user_msg_id = update.message.message_id

    # кнопки
    if user_text == "вернуться к выбору сигнала":
        await delete_previous_conversation(update, context)
        await update.message.reply_text("Возвращаюсь...", reply_markup=get_ghosts_keyboard(user_data))
        return GHOST_SELECTION

    if user_text == "подсказка":
        await delete_previous_conversation(update, context)
        hint = classifier_manager.get_hint(user_data, ghost_id)
        msg = await update.message.reply_text(f"*Подсказка:* {hint}", reply_markup=get_ghost_keyboard(ghost_id in passed))
        await save_current_conversation(user_data, current_user_msg_id, msg.message_id)
        return IN_GHOST

    if user_text == "история":
        await delete_previous_conversation(update, context)
        if ghost_id in passed or final_passed:
            mapping = user_data.get(USER_GHOST_RUNE_MAPPING, {})
            idx = mapping.get(ghost_id, 0)
            msg_text = story_manager.get_ghost_completion(ghost_id, idx)
            last_pw = get_last_used_password(ghost_id, user_data)
            if last_pw:
                first_pw = Config.GHOSTS[ghost_id]["passwords"][0]
                msg_text = msg_text.replace(f"\"{first_pw}\"", f"\"{last_pw}\"")
            msg = await update.message.reply_text(msg_text, reply_markup=get_ghost_keyboard(True))
        else:
            msg = await update.message.reply_text("История недоступна.", reply_markup=get_ghost_keyboard())
        await save_current_conversation(user_data, current_user_msg_id, msg.message_id)
        return IN_GHOST

    # если уже пройден (и финал не пройден)
    if ghost_id in passed and not final_passed:
        await delete_previous_conversation(update, context)
        msg = await update.message.reply_text(story_manager.get_silence_message(),
                                              reply_markup=get_ghost_keyboard(True))
        await save_current_conversation(user_data, current_user_msg_id, msg.message_id)
        return IN_GHOST

    # проверяем пароль
    password = get_current_password(ghost_id, user_data)

    if normalize_text(user_text) == normalize_text(password):
        await delete_previous_conversation(update, context)
        save_used_password(ghost_id, password, user_data)

        # добавляем в пройденные (если не финал)
        if not final_passed:
            passed.add(ghost_id)
            user_data[USER_PASSED_GHOSTS] = passed

            runes = user_data.get(USER_COLLECTED_RUNES, 0)
            if runes < 5:
                user_data[USER_COLLECTED_RUNES] = runes + 1

            # руна = индекс выбора
            order = user_data[USER_GHOSTS_ORDER]
            idx = order.index(ghost_id)
            mapping = user_data.get(USER_GHOST_RUNE_MAPPING, {})
            mapping[ghost_id] = idx
            user_data[USER_GHOST_RUNE_MAPPING] = mapping

        idx = user_data.get(USER_GHOST_RUNE_MAPPING, {}).get(ghost_id, 0)
        completion = story_manager.get_ghost_completion(ghost_id, idx)

        first_pw = Config.GHOSTS[ghost_id]["passwords"][0]
        completion = completion.replace(f"\"{first_pw}\"", f"\"{password}\"")

        if final_passed:
            assign_random_password(ghost_id, user_data)

        await update.message.reply_text(completion, reply_markup=get_ghost_keyboard(not final_passed))
        return IN_GHOST

    # классификатор в отдельном потоке
    malicious = await is_malicious_async(user_text, user_data, ghost_id)
    if malicious:
        await delete_previous_conversation(update, context)
        msg = await update.message.reply_text(
            classifier_manager.get_rejection_message(),
            reply_markup=get_ghost_keyboard()
        )
        await save_current_conversation(user_data, current_user_msg_id, msg.message_id)
        return IN_GHOST

    await delete_previous_conversation(update, context)

    loop = asyncio.get_event_loop()
    llm_task = loop.run_in_executor(None, llm_client.process_user_input, user_text, ghost_id, password)

    await typing_while_waiting(context, update.effective_chat.id, llm_task)

    answer = await llm_task

    msg = await update.message.reply_text(answer, reply_markup=get_ghost_keyboard())
    await save_current_conversation(user_data, current_user_msg_id, msg.message_id)

    return IN_GHOST


async def continue_final_part2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        Config.FINAL_MESSAGES["part2"],
        reply_markup=get_endings_keyboard()
    )
    return FINAL_PART2


async def handle_final_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get(USER_COLLECTED_RUNES, 0) < 5:
        await update.message.reply_text("Сначала соберите все руны.")
        return GHOST_SELECTION

    await update.message.reply_text(Config.FINAL_MESSAGES["part1"], reply_markup=get_continue_keyboard())
    return FINAL_PART1


async def handle_ending(update: Update, context: ContextTypes.DEFAULT_TYPE, num: int):
    ending = Config.FINAL_MESSAGES["endings"][f"ending{num}"]
    await update.message.reply_text(ending["messages"][0], reply_markup=get_continue_keyboard())
    context.user_data["pending_ending"] = ending
    return ENDING_PART1


async def handle_ending1(update, context):
    if update.message.text == Config.FINAL_MESSAGES["endings"]["ending1"]["name"]:
        return await handle_ending(update, context, 1)
    return FINAL_PART2


async def handle_ending2(update, context):
    if update.message.text == Config.FINAL_MESSAGES["endings"]["ending2"]["name"]:
        return await handle_ending(update, context, 2)
    return FINAL_PART2


async def handle_ending3(update, context):
    if update.message.text == Config.FINAL_MESSAGES["endings"]["ending3"]["name"]:
        return await handle_ending(update, context, 3)
    return FINAL_PART2


async def continue_ending_part2(update, context):
    ending = context.user_data.get("pending_ending")
    await update.message.reply_text(
        ending["messages"][1],
        reply_markup=get_remember_keyboard()
    )

    context.user_data[USER_FINAL_PASSED] = True
    context.user_data.pop("pending_ending", None)

    return ENDING_PART2


async def handle_remember(update, context):
    await update.message.reply_text(
        "Возвращаюсь к выбору призраков...",
        reply_markup=get_ghosts_keyboard(context.user_data)
    )
    return GHOST_SELECTION


def main():
    application = Application.builder().token(Config.BOT_TOKEN).concurrent_updates(True).build()

    ends = Config.FINAL_MESSAGES["endings"]
    e1 = ends["ending1"]["name"]
    e2 = ends["ending2"]["name"]
    e3 = ends["ending3"]["name"]

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            INTRO_PART1: [MessageHandler(filters.Regex("^Продолжить$"), continue_to_part2)],

            GHOST_SELECTION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ghost_selection)
            ],

            IN_GHOST: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ghost_interaction)
            ],

            FINAL_PART1: [
                MessageHandler(filters.Regex("^Продолжить$"), continue_final_part2)
            ],

            FINAL_PART2: [
                MessageHandler(filters.Regex(f"^{e1}$"), handle_ending1),
                MessageHandler(filters.Regex(f"^{e2}$"), handle_ending2),
                MessageHandler(filters.Regex(f"^{e3}$"), handle_ending3),
            ],

            ENDING_PART1: [
                MessageHandler(filters.Regex("^Продолжить$"), continue_ending_part2)
            ],

            ENDING_PART2: [
                MessageHandler(filters.Regex("^вспомнить былое$"), handle_remember)
            ],
        },
        fallbacks=[]
    )

    application.add_handler(conv)

    logger.info("Бот запущен (параллельный режим активен)…")
    application.run_polling()


if __name__ == "__main__":
    main()
