import psycopg
try:
    conn = psycopg.connect("postgresql://postgres:eve%40123@127.0.0.1:5432/spiritlink")
    print("CONNECTED OK (psycopg)")
    conn.close()
except Exception as e:
    print("ERROR:", type(e).__name__, e)
