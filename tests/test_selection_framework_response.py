import copy
from pathlib import Path
import sys
import unittest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from utils.llm_schemas import SelectionFrameworkResponse


class SelectionFrameworkTests(unittest.TestCase):
    def test_nested_selection_unchanged(self):
        data = {"selection": {"claims": ["Claim one", "Claim two"],
                              "framework": "Two pillars", "explanation": "Their reasoning"}}
        self.assertEqual(SelectionFrameworkResponse.model_validate(data).model_dump(), data)


    def test_flat_selection_preserves_all_values_and_input(self):
        data = {"selection": ["Claim one", "Claim two"],
                "framework": "Two pillars", "explanation": "Their reasoning"}
        original = copy.deepcopy(data)
        selection = SelectionFrameworkResponse.model_validate(data).selection
        self.assertEqual(selection.model_dump(), {"claims": original["selection"],
            "framework": original["framework"], "explanation": original["explanation"]})
        self.assertEqual(data, original)


    def test_missing_required_fields_still_rejected(self):
        for missing in ('framework', 'explanation'):
            with self.subTest(missing=missing):
                data = {"selection": ["Claim one"], "framework": "Structure", "explanation": "Reason"}
                del data[missing]
                with self.assertRaises(ValidationError):
                    SelectionFrameworkResponse.model_validate(data)


    def test_invalid_claims_are_not_replaced_or_invented(self):
        with self.assertRaises(ValidationError):
            SelectionFrameworkResponse.model_validate(
                {"selection": [None], "framework": "Structure", "explanation": "Reason"})


if __name__ == '__main__':
    unittest.main()
