import os
import pytest

os.environ["DB_PATH"] = ":memory:"

@pytest.fixture
def db():
    from db.database import Database
    database = Database(":memory:")
    database.initialize()
    yield database
    database.close()
