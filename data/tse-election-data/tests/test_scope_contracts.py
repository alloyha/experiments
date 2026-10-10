import unittest

from scripts.check_scope_contracts import validate_scope


class ScopeContractTests(unittest.TestCase):
    def test_valid_subset(self):
        self.assertEqual(
            validate_scope(
                {"2022", "2026"},
                {"general"},
                {"2026"},
                {"general"},
            ),
            [],
        )

    def test_incremental_year_must_be_selected(self):
        errors = validate_scope(
            {"2022"},
            {"general"},
            {"2026"},
            {"general"},
        )
        self.assertTrue(errors)

    def test_incremental_type_must_be_selected(self):
        errors = validate_scope(
            {"2026"},
            {"general"},
            {"2026"},
            {"municipal"},
        )
        self.assertTrue(errors)

    def test_year_type_calendar_mismatch_is_rejected(self):
        errors = validate_scope(
            {"2020"},
            {"general"},
            {"2020"},
            {"general"},
        )
        self.assertTrue(errors)

    def test_multiple_cycles_accept_required_types(self):
        self.assertEqual(
            validate_scope(
                {"2020", "2022", "2024", "2026"},
                {"general", "municipal"},
                {"2026"},
                {"general"},
            ),
            [],
        )

    def test_incremental_calendar_mismatch_is_rejected(self):
        errors = validate_scope(
            {"2020", "2022"},
            {"general", "municipal"},
            {"2020"},
            {"general"},
        )
        self.assertTrue(errors)

    def test_unknown_type_is_rejected(self):
        errors = validate_scope(
            {"2026"},
            {"general", "unknown"},
            {"2026"},
            {"general"},
        )
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
