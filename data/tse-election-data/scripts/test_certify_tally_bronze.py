import unittest
from pathlib import Path
from certify_tally_bronze import certified_expression, identifier, literal

class QuerySafetyTests(unittest.TestCase):
    def test_escape_identifier(self):
        self.assertEqual(identifier('a"b'), '"a""b"')

    def test_escape_literal(self):
        self.assertEqual(literal("a'b"), "'a''b'")

    def test_file_root_boundary(self):
        sql = certified_expression('source_file', 'r', Path('/tmp/tse-c3-recovery'))
        self.assertIn("'/tmp/tse-c3-recovery/'", sql)
        self.assertIn('starts_with', sql)

    def test_measure_unchanged(self):
        self.assertEqual(certified_expression('valid_votes', 'r', Path('/tmp')),
                         'r."valid_votes"')

if __name__ == '__main__':
    unittest.main()
