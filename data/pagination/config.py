import logging

# Constants
TOTAL_ROWS = 100_000
LIMIT = 1_000
BATCH_SIZE = 1_000

# Database configuration
DB_CONFIG = {
    "dbname": "pagination_db",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5432"
}

# Setting up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()