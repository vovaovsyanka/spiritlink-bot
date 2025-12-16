# SpiritLink Telegram Bot

Бот в стиле Gandalf AI для изучения промпт-инъекций через узнавание защищенных секретных слов с системой уровней и сюжетом.

## Описание

Пользователь пытается узнать "Слово-Якорь" через prompt-инъекции, а бот защищает его с помощью классификаторов и дает ответы с помощью LLM.

## Установка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/ваш-username/ваш-репозиторий.git
```

2. Установите зависимости:

```bash
pip install -r requirements.txt
```

3. Создайте переменные окружения:
BOT_TOKEN=your_telegram_bot_token
OLLAMA_API_KEY=your_ollama_api_key
OLLAMA_API_KEY2=your_backup_ollama_api_key

4. Создайте БД
Для этого:
- установите PostgreSQL 18
- во время установки создайте суперпользователя (рекомендуется postgres:eve@123)
- добавьте `C:\Program Files\PostgreSQL\18\bin\` в PATH
- создайте базу spiritlink и таблицы в ней:
    
```bash
psql -f db/schema.sql -U postgres spiritlink
```

- назначьте суперпользователю доступ

```bash
psql -h 127.0.0.1 -p 5432 -U postgres spiritlink

GRANT SELECT, INSERT, UPDATE ON TABLE users TO postgres;
```

5. Запустите бота:
```bash
python main.py
```
