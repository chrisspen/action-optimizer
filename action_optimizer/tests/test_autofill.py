# action_optimizer/tests/test_autofill.py
import unittest
import tempfile
from pathlib import Path

import ezodf

from action_optimizer.autofill import autofill_ods


class Tests(unittest.TestCase):

    def test_autofill(self):
        """
        Test that empty cells above the first non-empty value get filled with 'last' logic.
        """
        # 1. Create minimal ODS
        doc = ezodf.newdoc(doctype="ods")
        sheet = doc.sheets[0]

        # Header row (0)
        sheet[0, 0].set_value("Date")
        sheet[0, 1].set_value("Test Col")

        # Default row (11)
        sheet[11, 0].set_value("default")
        sheet[11, 1].set_value("last")

        # First data row (12) - only this has a value
        sheet[12, 0].set_value("2025-10-28")
        sheet[12, 1].set_value(42) # Numeric value to fill upwards

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".ods", delete=False) as tmp:
            doc.saveas(tmp.name)
        input_path = Path(tmp.name)

        # 2. Run autofill
        output_path = input_path.with_stem(input_path.stem + "_autofilled")
        autofill_ods(input_path, output_path)

        # 3. Verify output
        result_doc = ezodf.opendoc(str(output_path))
        result_sheet = result_doc.sheets[0]

        # Headers/default untouched
        self.assertEqual(result_sheet[0, 1].value, "Test Col")
        self.assertEqual(result_sheet[11, 1].value, "last")

        # Filled value in row 12 (original)
        self.assertEqual(result_sheet[12, 1].value, 42)

        # Clean up files
        input_path.unlink()
        output_path.unlink()
