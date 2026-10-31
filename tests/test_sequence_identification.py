import unittest

from preprocess.convert import identify_sequence


class TestSequenceIdentification(unittest.TestCase):
    def test_adc_is_checked_before_dwi(self):
        self.assertEqual(identify_sequence("ep2d_diff_b50_800_ADC"), "ADC")

    def test_dwi_sequence(self):
        self.assertEqual(identify_sequence("ep2d_diff_b50_800_TRACEW"), "DWI")

    def test_dixon_sequences(self):
        self.assertEqual(identify_sequence("new_t1_dixon_W"), "DixonW")
        self.assertEqual(identify_sequence("new_t1_dixon_F"), "DixonF")
        self.assertEqual(identify_sequence("new_t1_dixon_in"), "DixonIP")
        self.assertEqual(identify_sequence("new_t1_dixon_opp"), "DixonOP")

    def test_excluded_sequences(self):
        self.assertIsNone(identify_sequence("localizer"))
        self.assertIsNone(identify_sequence("MIP"))


if __name__ == "__main__":
    unittest.main()
