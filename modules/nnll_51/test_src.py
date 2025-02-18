import sqlite3
import os
from contextlib import closing
import unittest
from sqlite3 import ProgrammingError
from modules.nnll_51.src import managed_connection, create_table, insert_into_db


class TestDatabaseOperations(unittest.TestCase):
    DATABASE_NAME = "test_combo_history.db"

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

    @classmethod
    def setUpClass(cls):
        with managed_connection(cls.DATABASE_NAME) as conn:
            create_table(conn)

    def test_insert_and_retrieve(self):
        user_set = {
            "noise_seed": 42,
            "guidance_scale": 7.5,
            "width": 1024,
            "height": 768,
            "safety_checker": True,
            "output_type": "jpeg",
        }
        with managed_connection(self.DATABASE_NAME) as conn:
            insert_into_db(conn, user_set)

            cursor = conn.cursor()
            cursor.execute("SELECT * FROM parameters")
            result = cursor.fetchone()

        print(result)
        self.assertIsNotNone(result)
        self.assertEqual(result[7], 42)
        self.assertAlmostEqual(result[9], 7.5)
        self.assertEqual(result[10], 1024)
        self.assertEqual(result[11], 768)
        self.assertTrue(result[13])
        self.assertEqual(result[19], "jpeg")

    def tearDown(self):
        try:
            os.remove(self.test_db)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    unittest.main()
