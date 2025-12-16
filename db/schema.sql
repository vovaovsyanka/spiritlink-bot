-- users: общая информация о пользователе
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    telegram_id BIGINT UNIQUE NOT NULL,

    ghosts_order INTEGER[] NOT NULL DEFAULT '{}',         -- порядок выбора призраков
    passed_ghosts INTEGER[] NOT NULL DEFAULT '{}',        -- пройденные призраки
    collected_runes INTEGER NOT NULL DEFAULT 0,
    ghost_rune_mapping JSONB NOT NULL DEFAULT '{}' ,      -- {"1":0, "3":2} или { "1": 0, ... }
    final_passed BOOLEAN NOT NULL DEFAULT FALSE,

    current_ghost INTEGER NULL,                           -- текущий призрак (id) или NULL

    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_users_telegram_id ON users (telegram_id);

-- user_ghosts: per-user-per-ghost storage (пароли, подсказки, использованные пароли и подсказки)
CREATE TABLE IF NOT EXISTS user_ghosts (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ghost_id INTEGER NOT NULL,

    current_password TEXT,
    last_used_password TEXT,
    used_passwords TEXT[] NOT NULL DEFAULT '{}',   -- список прошлых паролей (text[])
    used_hints JSONB NOT NULL DEFAULT '{}' ,       -- структура: { "Dictionary": [...], "RuBert": [...] } или любая json

    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),

    PRIMARY KEY (user_id, ghost_id)
);

CREATE INDEX IF NOT EXISTS idx_user_ghosts_user_id ON user_ghosts (user_id);
