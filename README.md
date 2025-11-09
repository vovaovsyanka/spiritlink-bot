# SpiritLink Telegram Bot

Бот в стиле Gandalf AI для защиты секретных слов через систему уровней.

## Описание

Пользователь пытается узнать "Слово-Якорь" через prompt-инъекции, а бот защищает его с помощью классификаторов и LLM.

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/ваш-username/ваш-репозиторий.git

2. Установите зависимости:
```bash
pip install -r requirements.txt

3. Настройте переменные окружения в файле .env:
BOT_TOKEN=your_telegram_bot_token
OLLAMA_API_KEY=your_ollama_api_key
OLLAMA_API_KEY2=your_backup_ollama_api_key

4. Запустите бота:
```bash
python main.py