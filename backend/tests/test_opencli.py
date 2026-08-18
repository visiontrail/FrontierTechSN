import unittest

from backend.pipeline.opencli import OpenCLIError, first_json


class OpenCLIOutputTests(unittest.TestCase):
    def test_first_json_recovers_cli_prose_and_fenced_payload(self):
        value = 'Gemini response:\n```json\n{"start_seconds": 4, "end_seconds": 10}\n```'

        self.assertEqual(first_json(value)["start_seconds"], 4)

    def test_first_json_rejects_non_json_output(self):
        with self.assertRaises(OpenCLIError):
            first_json("browser returned no structured payload")


if __name__ == "__main__":
    unittest.main()
