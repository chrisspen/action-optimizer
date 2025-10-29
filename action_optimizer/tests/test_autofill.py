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

    def test_autofill_with_repeated_rows(self):
        """
        Ensure default extraction works even when header rows rely on row repetition.
        """
        doc = ezodf.newdoc(doctype="ods")
        sheet = ezodf.Table("Sheet1", size=(40, 5))
        doc.sheets += sheet

        # Header row
        headers = ["key", "metric", "amount", "note", "extra"]
        for idx, name in enumerate(headers):
            sheet[0, idx].set_value(name)

        # Meta row that should remain untouched
        sheet[1, 0].set_value("meta-key")
        sheet[1, 1].set_value("meta-value")

        # Create a block of repeated blank rows (simulates wide sheet metadata spacing)
        blank_row_node = sheet.xmlnode.findall(
            "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}table-row"
        )[2]
        blank_row_node.set(
            "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}number-rows-repeated", "8"
        )

        # Default row positioned after the repeated block
        default_idx = 12
        sheet[default_idx, 0].set_value("default")
        sheet[default_idx, 1].set_value("last")
        sheet[default_idx, 2].set_value("5")

        # First populated data row further down
        data_idx = 20
        sheet[data_idx, 1].set_value(42)
        sheet[data_idx, 2].set_value(7)

        with tempfile.NamedTemporaryFile(suffix=".ods", delete=False) as tmp_in:
            doc.saveas(tmp_in.name)
            input_path = Path(tmp_in.name)

        output_path = input_path.with_stem(input_path.stem + "_autofilled")

        try:
            baseline_doc = ezodf.opendoc(str(input_path))
            baseline_sheet = baseline_doc.sheets[0]
            default_row = None
            first_data_row = None
            baseline_default_metric = None
            baseline_default_amount = None
            for idx in range(baseline_sheet.nrows()):
                cell_value = baseline_sheet[idx, 0].value
                if (
                    default_row is None
                    and cell_value is not None
                    and str(cell_value).strip().lower() == "default"
                ):
                    default_row = idx
                metric_value = baseline_sheet[idx, 1].value
                if (
                    default_row is not None
                    and first_data_row is None
                    and metric_value not in (None, "", " ")
                    and idx > default_row
                ):
                    first_data_row = idx
                    break
            if default_row is not None:
                baseline_default_metric = baseline_sheet[default_row, 1].value
                baseline_default_amount = baseline_sheet[default_row, 2].value
            self.assertIsNotNone(default_row)
            self.assertIsNotNone(first_data_row)

            autofill_ods(input_path, output_path)

            result_doc = ezodf.opendoc(str(output_path))
            result_sheet = result_doc.sheets[0]

            # Meta header row should be unchanged
            self.assertEqual(result_sheet[1, 0].value, "meta-key")
            self.assertEqual(result_sheet[1, 1].value, "meta-value")

            # Default row values remain intact
            result_default_row = None
            for idx in range(result_sheet.nrows()):
                if (
                    result_sheet[idx, 0].value is not None
                    and str(result_sheet[idx, 0].value).strip().lower() == "default"
                ):
                    result_default_row = idx
                    break
            self.assertIsNotNone(result_default_row)
            if baseline_default_metric is not None:
                self.assertEqual(result_sheet[result_default_row, 1].value, baseline_default_metric)
            if baseline_default_amount is not None:
                self.assertEqual(result_sheet[result_default_row, 2].value, baseline_default_amount)

            # Locate rows dynamically since repeated rows expand on save.
            # Rows between default and first data row should mirror the first data row.
            if first_data_row - default_row > 1:
                fill_row = first_data_row - 1
                self.assertEqual(result_sheet[fill_row, 1].value, 42.0)
                self.assertEqual(result_sheet[fill_row, 2].value, 5.0)

            # First data row retains its original values
            self.assertEqual(result_sheet[first_data_row, 1].value, 42.0)
            self.assertEqual(result_sheet[first_data_row, 2].value, 7.0)
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
