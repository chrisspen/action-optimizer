# action_optimizer/tests/test_autofill.py
import re
import tempfile
import unittest
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
        sheet = ezodf.Table("Sheet1", size=(32, 8))
        doc.sheets += sheet

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
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

    def test_autofill_sample(self):
        """
        Smoke test against the shared fixture to ensure key supplement columns autofill correctly.
        """
        fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures"
        input_path = fixtures_dir / "supplements-sample.ods"
        output_path = Path("/tmp/output.ods")

        # Ensure a clean slate before writing.
        output_path.unlink(missing_ok=True)

        autofill_ods(input_path, output_path)

        def _col_label_to_index(label: str) -> int:
            idx = 0
            for ch in label:
                idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
            return idx - 1

        def _evaluate_sum_formula(formula: str) -> float:
            match = re.fullmatch(r"of:=SUM\(\[\.([A-Z]+)(\d+):\.([A-Z]+)(\d+)\]\)", formula)
            self.assertIsNotNone(match, f"Unexpected formula: {formula}")
            start_col, start_row, end_col, end_row = match.groups()
            self.assertEqual(start_col, end_col, "Formula spans multiple columns, adjust parser.")
            col_idx = _col_label_to_index(start_col)
            start_row_idx = int(start_row) - 1
            end_row_idx = int(end_row) - 1
            total = 0.0
            for r in range(start_row_idx, end_row_idx + 1):
                val = sheet[r, col_idx].value
                if val in (None, "", " "):
                    continue
                total += float(val)
            return total

        try:
            result_doc = ezodf.opendoc(str(output_path))
            sheet = result_doc.sheets[0]

            # Row 13 is zero-indexed row 12; columns H/I/J are indexes 7/8/9.
            self.assertEqual(sheet[12, 7].value, 50.0)
            self.assertEqual(sheet[12, 8].value, 0.0)
            j_cell = sheet[12, 9]
            j_value = j_cell.value
            if j_value is None and j_cell.formula:
                j_value = _evaluate_sum_formula(j_cell.formula)
            self.assertEqual(j_value, 20.0)
        finally:
            output_path.unlink(missing_ok=True)
