import os


required_env = [
    "DB_TABLE",
    "DB_DRIVERNAME",
    "DB_HOST",
    "DB_PORT",
    "DB_USERNAME",
    "DB_PASSWORD",
    "DB_DATABASE",
]

for env in required_env:
    assert os.getenv(env)

table = os.getenv("DB_TABLE")

DATABASE = {
    "drivername": os.getenv("DB_DRIVERNAME"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "username": os.getenv("DB_USERNAME"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_DATABASE"),
}
