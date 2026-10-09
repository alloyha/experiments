import unittest
from pathlib import Path
class PreparedElectorateTests(unittest.TestCase):
    def test_bronze_supports_prepared_and_safe_fallback(self):
        root=Path(__file__).resolve().parents[1]
        text=(root/'tse_dbt/models/bronze/electorate/bronze_electorate.sql').read_text()
        for token in ('use_prepared_electorate','electorate_prepared_root','read_parquet','strict_mode=false','null_padding=true','parallel=false'):
            self.assertIn(token,text)
if __name__=='__main__': unittest.main()
