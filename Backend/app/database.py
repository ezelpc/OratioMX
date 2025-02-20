import psycopg2
from config import CONFIG

def get_db_connection():
    return psycopg2.connect(
        dbname=CONFIG['dbname'],
        user=CONFIG['user'],
        password=CONFIG['password'],
        host=CONFIG['host'],
        port=CONFIG['port']
    )
