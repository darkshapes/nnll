import sqlite3
import os
from contextlib import closing
import unittest
from sqlite3 import ProgrammingError
from modules.nnll_51.src import managed_connection, create_table


class TestDatabaseOperations(unittest.TestCase):
    def setUp(self):
        self.test_db = "test_db.db"

    def test_managed_connection(self):
        with managed_connection(self.test_db) as conn:
            self.assertIsNotNone(conn, "Connection should be established.")
        # Assert connection is closed
        with self.assertRaises(ProgrammingError):
            conn.cursor()

    def test_create_table(self):
        with managed_connection(self.test_db) as conn:
            create_table(conn)
        with closing(sqlite3.connect(self.test_db)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='parameters'")
            self.assertIsNotNone(cursor.fetchone(), "Table 'parameters' should exist.")

    def tearDown(self):
        try:
            os.remove(self.test_db)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    unittest.main()
