import unittest
from datetime import date

from modeling.nifty_universe import Membership, local_symbol, members_on, parse_nifty_membership


class NiftyUniverseTests(unittest.TestCase):
    def test_ltim_alias_is_limited_to_verified_membership_interval(self):
        intervals = (Membership("LTM", date(2023, 7, 13), date(2024, 9, 30), "test", ""),)
        self.assertEqual(local_symbol("LTM", intervals), "LTIM")
        unsafe = (Membership("LTM", date(2022, 1, 1), None, "test", ""),)
        with self.assertRaises(ValueError):
            local_symbol("LTM", unsafe)

    def test_membership_uses_half_open_intervals(self):
        data = (
            "index_name,symbol,valid_from,valid_to,source,source_url\n"
            "Nifty 50,AAA,2020-01-01,2022-04-01,circular,https://example.com/a\n"
            "Nifty 50,BBB,2022-04-01,,circular,https://example.com/b\n"
            "Nifty 100,CCC,2020-01-01,,snapshot,\n"
        )
        intervals = parse_nifty_membership(data)
        self.assertEqual(members_on(intervals, date(2022, 3, 31)), {"AAA"})
        self.assertEqual(members_on(intervals, date(2022, 4, 1)), {"BBB"})

    def test_rejects_unverified_dates_and_malformed_data(self):
        data = (
            "index_name,symbol,valid_from,valid_to,source,source_url\n"
            "Nifty 50,AAA,2020-01-01,,snapshot,\n"
        )
        intervals = parse_nifty_membership(data)
        with self.assertRaises(ValueError):
            members_on(intervals, date(2018, 3, 31))
        with self.assertRaises(ValueError):
            members_on(intervals, date(2026, 10, 14))
        with self.assertRaises(ValueError):
            parse_nifty_membership("index_name,symbol\nNifty 50,AAA\n")


if __name__ == "__main__":
    unittest.main()
