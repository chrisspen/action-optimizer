# action_optimizer/tests/test_autofill.py
import tempfile
import unittest
from pathlib import Path

from odf.namespaces import OFFICENS, TABLENS
from odf.opendocument import OpenDocumentSpreadsheet
from odf.table import Table, TableCell, TableRow
from odf.text import P

from action_optimizer.autofill import autofill_ods, read_row_from_zip
from action_optimizer.ods import OdsDocument


def _set_element_text(element, text: str | None) -> None:
    """Replace textual content for tests when constructing fixtures."""
    while element.hasChildNodes():
        element.removeChild(element.firstChild)
    if text:
        paragraph = P()
        paragraph.addText(text)
        element.addElement(paragraph)


def _set_cell_value(cell: TableCell, value) -> None:
    if value is None:
        _set_element_text(cell, None)
        return
    if isinstance(value, (int, float)):
        cell.setAttrNS(OFFICENS, "value-type", "float")
        cell.setAttrNS(OFFICENS, "value", str(value))
        if cell.getAttrNS(OFFICENS, "string-value") is not None:
            cell.removeAttrNS(OFFICENS, "string-value")
    else:
        cell.setAttrNS(OFFICENS, "value-type", "string")
        cell.setAttrNS(OFFICENS, "string-value", str(value))
        if cell.getAttrNS(OFFICENS, "value") is not None:
            cell.removeAttrNS(OFFICENS, "value")
    _set_element_text(cell, str(value))


class Tests(unittest.TestCase):

    def test_autofill(self):
        """
        Test that empty cells above the first non-empty value get filled with 'last' logic.
        """
        doc = OdsDocument.new_blank("Sheet1", 32, 8)
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

        with tempfile.NamedTemporaryFile(suffix=".ods", delete=False) as tmp:
            doc.save(tmp.name)
            input_path = Path(tmp.name)

        output_path = input_path.with_stem(input_path.stem + "_autofilled")
        autofill_ods(input_path, output_path)

        result_doc = OdsDocument.load(str(output_path))
        result_sheet = result_doc.sheets[0]

        self.assertEqual(result_sheet[0, 1].value, "Test Col")
        self.assertEqual(result_sheet[11, 1].value, "last")
        self.assertEqual(result_sheet[12, 1].value, 42.0)

        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

    def test_autofill_sample(self):
        """
        Smoke test against the shared fixture to ensure key supplement columns autofill correctly.
        """
        fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures"
        input_path = fixtures_dir / "supplements-sample.ods"
        output_path = Path("/tmp/output.ods")
        output_path.unlink(missing_ok=True)

        autofill_ods(input_path, output_path)

        try:
            result_doc = OdsDocument.load(str(output_path))
            sheet = result_doc.sheets[0]

            self.assertEqual(sheet[12, 7].value, 50.0)
            self.assertEqual(sheet[12, 8].value, 0.0)
            j_cell = sheet[12, 9]
            self.assertEqual(j_cell.formula, "of:=SUM([.I13:.I23])")
            self.assertEqual(j_cell.value, 0.0)
        finally:
            output_path.unlink(missing_ok=True)

    def test_absolute_row_reference_preserved(self):
        """
        Ensure formulas keep absolute row references (`$`) when shifted upward.
        """
        doc = OdsDocument.new_blank("Sheet1", 24, 6)
        sheet = doc.sheets[0]

        # Headers and defaults
        sheet[0, 0].set_value("Date")
        sheet[0, 2].set_value("next_ex")
        sheet[11, 0].set_value("default")
        sheet[11, 2].set_value("last")

        base_formula = "of:=INDEX([.C$1:.E$1]; MATCH(MAX([.C15:.E15]); [.C15:.E15]; 0))"
        sheet[15, 0].set_value("2025-01-15")
        sheet[15, 2].formula = base_formula
        sheet[15, 2].set_value("cached")

        with tempfile.NamedTemporaryFile(suffix=".ods", delete=False) as tmp_in:
            doc.save(tmp_in.name)
            input_path = Path(tmp_in.name)

        output_path = input_path.with_stem(input_path.stem + "_autofilled")

        try:
            autofill_ods(input_path, output_path)

            result_doc = OdsDocument.load(str(output_path))
            result_sheet = result_doc.sheets[0]
            filled_formula = result_sheet[12, 2].formula

            expected_formula = "of:=INDEX([.C$1:.E$1]; MATCH(MAX([.C12:.E12]); [.C12:.E12]; 0))"
            self.assertEqual(filled_formula, expected_formula)
            self.assertEqual(result_sheet[12, 2].value, "cached")
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_named_formula_with_digits_preserved(self):
        """
        Ensure formulas referencing named ranges with digits are not altered.
        """
        doc = OdsDocument.new_blank("Sheet1", 24, 4)
        sheet = doc.sheets[0]

        sheet[0, 0].set_value("Date")
        sheet[0, 1].set_value("dose_formula")

        sheet[11, 0].set_value("default")
        sheet[11, 1].set_value("last")

        base_formula = "of:=EV_D3_IU"
        sheet[15, 0].set_value("2025-01-15")
        sheet[15, 1].formula = base_formula
        sheet[15, 1].set_value(88)

        with tempfile.NamedTemporaryFile(suffix=".ods", delete=False) as tmp_in:
            doc.save(tmp_in.name)
            input_path = Path(tmp_in.name)

        output_path = input_path.with_stem(input_path.stem + "_autofilled")

        try:
            autofill_ods(input_path, output_path)

            result_doc = OdsDocument.load(str(output_path))
            result_sheet = result_doc.sheets[0]
            filled_cell = result_sheet[12, 1]

            self.assertEqual(filled_cell.formula, base_formula)
            self.assertEqual(filled_cell.value, 88.0)
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_header_row_is_preserved(self):
        """
        Ensure autofill preserves metadata in the protected header rows.
        """
        doc = OdsDocument.new_blank("Sheet1", 16, 3)
        sheet = doc.sheets[0]

        # Header row values.
        sheet[0, 0].set_value("key")
        sheet[0, 1].set_value("learn")
        sheet[0, 2].set_value("predict")

        sheet[1, 0].set_value("numeric")
        sheet[2, 0].set_value("0,60,15")
        sheet[8, 0].set_value("1")
        sheet[9, 0].set_value("1")

        sheet[11, 0].set_value("default")
        sheet[11, 1].set_value("last")

        sheet[12, 0].set_value("2025-11-02")
        sheet[12, 1].set_value(5)

        with tempfile.NamedTemporaryFile(suffix=".ods", delete=False) as tmp_in:
            doc.save(tmp_in.name)
            input_path = Path(tmp_in.name)

        output_path = input_path.with_stem(input_path.stem + "_autofilled")

        try:
            # Capture metadata before running autofill.
            baseline_learn = read_row_from_zip(input_path, "Sheet1", 8, 3)[1]
            baseline_predict = read_row_from_zip(input_path, "Sheet1", 9, 3)[1]

            autofill_ods(input_path, output_path)

            header_learn = read_row_from_zip(output_path, "Sheet1", 8, 3)[1]
            header_predict = read_row_from_zip(output_path, "Sheet1", 9, 3)[1]

            self.assertEqual(baseline_learn, header_learn)
            self.assertEqual(baseline_predict, header_predict)
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_autofill_with_repeated_rows(self):
        """
        Ensure default extraction works even when header rows rely on row repetition.
        """

        def add_row(table: Table, values, repeat: int = 1):
            row = TableRow()
            for value in values:
                cell = TableCell()
                if value is not None:
                    _set_cell_value(cell, value)
                row.addElement(cell)
            if repeat > 1:
                row.setAttrNS(TABLENS, "number-rows-repeated", str(repeat))
            table.addElement(row)
            return row

        doc = OpenDocumentSpreadsheet()
        table = Table(name="Sheet1")
        doc.spreadsheet.addElement(table)

        add_row(table, ["key", "metric", "amount", "note", "extra"]) # header row 0
        add_row(table, ["meta-key", "meta-value", None, None, None]) # meta row 1
        add_row(table, [None, None, None, None, None], repeat=8) # repeated block rows 2-9
        add_row(table, [None, None, None, None, None]) # row 10
        add_row(table, [None, None, None, None, None]) # row 11
        add_row(table, ["default", "last", "5", None, None]) # default row 12
        for _ in range(7): # rows 13-19 blank
            add_row(table, [None, None, None, None, None])
        add_row(table, [None, 42, 7, None, None]) # data row 20

        with tempfile.NamedTemporaryFile(suffix=".ods", delete=False) as tmp_in:
            doc.save(tmp_in.name)
            input_path = Path(tmp_in.name)

        output_path = input_path.with_stem(input_path.stem + "_autofilled")

        try:
            baseline_doc = OdsDocument.load(str(input_path))
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

            result_doc = OdsDocument.load(str(output_path))
            result_sheet = result_doc.sheets[0]

            self.assertEqual(result_sheet[1, 0].value, "meta-key")
            self.assertEqual(result_sheet[1, 1].value, "meta-value")

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

            if first_data_row - default_row > 1:
                fill_row = first_data_row - 1
                self.assertEqual(result_sheet[fill_row, 1].value, 42.0)
                self.assertEqual(result_sheet[fill_row, 2].value, 5.0)

            self.assertEqual(result_sheet[first_data_row, 1].value, 42.0)
            self.assertEqual(result_sheet[first_data_row, 2].value, 7.0)
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
