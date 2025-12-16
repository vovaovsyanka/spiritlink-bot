# db.py
import asyncpg
import json
from typing import Dict, Any
from pathlib import Path
from states import (
    USER_GHOSTS_ORDER, USER_PASSED_GHOSTS, USER_CURRENT_GHOST,
    USER_COLLECTED_RUNES, USER_FINAL_PASSED, USER_GHOST_RUNE_MAPPING
)
from config import Config
import logging

logger = logging.getLogger(__name__)

_pool: asyncpg.pool.Pool = None

async def init_db_pool():
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn=Config.DB_DSN, min_size=1, max_size=10)
        logger.info("Postgres pool opened")
    return _pool

async def close_db_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Postgres pool closed")

async def _ensure_user(conn, telegram_id: int):
    row = await conn.fetchrow("SELECT id FROM users WHERE telegram_id = $1", telegram_id)
    if row:
        return row['id']
    rec = await conn.fetchrow(
        "INSERT INTO users (telegram_id) VALUES ($1) RETURNING id", telegram_id
    )
    return rec['id']

async def load_user_data(telegram_id: int) -> Dict[str, Any]:
    """
    Загружает данные пользователя из БД и возвращает dict, совместимый с context.user_data.
    Если пользователя нет — создаёт новую запись с дефолтными значениями.
    """
    pool = await init_db_pool()
    async with pool.acquire() as conn:
        user_id = await _ensure_user(conn, telegram_id)

        user_row = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        # преобразуем
        user_data = {}

        # ghosts_order: integer[] -> list
        user_data[USER_GHOSTS_ORDER] = list(user_row['ghosts_order']) if user_row['ghosts_order'] else []
        # passed_ghosts: integer[] -> set
        user_data[USER_PASSED_GHOSTS] = set(user_row['passed_ghosts']) if user_row['passed_ghosts'] else set()
        user_data[USER_COLLECTED_RUNES] = user_row['collected_runes'] or 0
        user_data[USER_FINAL_PASSED] = bool(user_row['final_passed'])
        # user_data[USER_GHOST_RUNE_MAPPING] = dict(user_row['ghost_rune_mapping']) if user_row['ghost_rune_mapping'] else {}
        mapping = user_row['ghost_rune_mapping']

        user_data[USER_GHOST_RUNE_MAPPING] = {}

        if mapping:
            if isinstance(mapping, dict):
                user_data[USER_GHOST_RUNE_MAPPING] = mapping
            elif isinstance(mapping, str):
                import json
                user_data[USER_GHOST_RUNE_MAPPING] = json.loads(mapping)


        user_data[USER_CURRENT_GHOST] = user_row['current_ghost']

        # Load per-ghost rows
        rows = await conn.fetch("SELECT * FROM user_ghosts WHERE user_id = $1", user_id)
        # Build used_hints global structure: user_data['used_hints'][ghost_id] = ... (matches ClassifierManager expectation)
        used_hints_global = {}
        for r in rows:
            gid = r['ghost_id']
            # current_password / last_used_password / used_passwords
            if r['current_password'] is not None:
                user_data[f'ghost_{gid}_password'] = r['current_password']
            if r['last_used_password'] is not None:
                user_data[f'ghost_{gid}_last_used_password'] = r['last_used_password']
            if r['used_passwords'] is not None:
                user_data[f'ghost_{gid}_used_passwords'] = list(r['used_passwords'])
            # used_hints JSONB
            hints = r['used_hints'] or {}
            # ensure structure user_data['used_hints'][str(ghost_id)] = hints
            used_hints_global[str(gid)] = hints

        if used_hints_global:
            user_data['used_hints'] = used_hints_global

        return user_data

async def save_user_data(telegram_id: int, user_data: Dict[str, Any]):
    """
    Сохраняет user_data в БД. Делает UPSERT для users и upsert для user_ghosts, в зависимости от наличия данных.
    """
    pool = await init_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # ensure user exists
            row = await conn.fetchrow("SELECT id FROM users WHERE telegram_id = $1", telegram_id)
            if row:
                user_id = row['id']
            else:
                rec = await conn.fetchrow("INSERT INTO users (telegram_id) VALUES ($1) RETURNING id", telegram_id)
                user_id = rec['id']

            # Prepare columns for users
            ghosts_order = user_data.get(USER_GHOSTS_ORDER, [])
            passed_ghosts = list(user_data.get(USER_PASSED_GHOSTS, set())) if user_data.get(USER_PASSED_GHOSTS) is not None else []
            collected_runes = int(user_data.get(USER_COLLECTED_RUNES, 0))
            final_passed = bool(user_data.get(USER_FINAL_PASSED, False))
            ghost_rune_mapping = user_data.get(USER_GHOST_RUNE_MAPPING, {})

            await conn.execute(
                """
                UPDATE users SET
                    ghosts_order = $1,
                    passed_ghosts = $2,
                    collected_runes = $3,
                    ghost_rune_mapping = $4,
                    final_passed = $5,
                    current_ghost = $6,
                    updated_at = now()
                WHERE id = $7
                """,
                ghosts_order, passed_ghosts, collected_runes, json.dumps(ghost_rune_mapping), final_passed,
                user_data.get(USER_CURRENT_GHOST), user_id
            )

            # Перебираем призраков 1..5 (или любые, которые есть в данных)
            # Собираем список ghost_id, для которых есть ghost_* ключи или used_hints
            ghost_ids = set()
            # По ключам ghost_{id}_...
            for k in list(user_data.keys()):
                if k.startswith('ghost_') and k.count('_') >= 2:
                    parts = k.split('_')
                    try:
                        gid = int(parts[1])
                        ghost_ids.add(gid)
                    except:
                        pass
            # Также возьмём из ghosts_order
            for gid in user_data.get(USER_GHOSTS_ORDER, []):
                ghost_ids.add(gid)
            # Ограничьте, если нужно, набором существующих призраков (например 1..5)
            # ghost_ids = {g for g in ghost_ids if 1 <= g <= 5}

            # used_hints может иметь entries like {'1': {...}, '2': {...}}
            used_hints_global = user_data.get('used_hints', {})

            for gid in ghost_ids:
                current_password = user_data.get(f'ghost_{gid}_password')
                last_used = user_data.get(f'ghost_{gid}_last_used_password')
                used_passwords = user_data.get(f'ghost_{gid}_used_passwords', [])

                hints_for_ghost = used_hints_global.get(str(gid)) if isinstance(used_hints_global, dict) else None
                if hints_for_ghost is None:
                    hints_for_ghost = {}

                # UPSERT into user_ghosts
                await conn.execute(
                    """
                    INSERT INTO user_ghosts (user_id, ghost_id, current_password, last_used_password, used_passwords, used_hints, created_at, updated_at)
                    VALUES ($1,$2,$3,$4,$5,$6, now(), now())
                    ON CONFLICT (user_id, ghost_id) DO UPDATE
                      SET current_password = EXCLUDED.current_password,
                          last_used_password = EXCLUDED.last_used_password,
                          used_passwords = EXCLUDED.used_passwords,
                          used_hints = EXCLUDED.used_hints,
                          updated_at = now()
                    """,
                    user_id, gid, current_password, last_used, used_passwords or [], json.dumps(hints_for_ghost)
                )
