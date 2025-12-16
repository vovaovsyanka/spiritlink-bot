# test_db_async.py
import asyncio
import asyncpg

async def main():
    try:
        conn = await asyncpg.connect(
            user="postgres",
            password="eve@123",
            database="spiritlink",
            host="127.0.0.1",
            port=5432,
            ssl=False
        )
        print("CONNECTED OK (asyncpg)")
        await conn.close()
    except Exception as e:
        print("ERROR:", type(e).__name__, e)

if __name__ == "__main__":
    asyncio.run(main())
