#!/usr/bin/env python3
"""
Autofill .ODS spreadsheet columns based on a 'default' row.

Usage:
  python autofill.py input.ods [--output result.ods]
"""

import argparse
import re
import sys
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

from action_optimizer.ods import OdsDocument

# ----------------------------------------------------------------------
# Helper: shift ODS cell references (A1, $A$1, .A1, etc.) by a row offset
# ----------------------------------------------------------------------
_CELL_REF_RE = re.compile(r"([.$]?)([A-Z]+)([.$]?)(\d+)")


def _shift_ref(match, row_offset):
    abs_col, col, abs_row, row = match.groups()
    new_row = int(row) + row_offset
    return f"{abs_col}{col}{abs_row}{new_row}"


def adjust_formula(formula: str, row_offset: int) -> str:
    """Shift all row numbers in an ODS formula by *row_offset*."""
    return _CELL_REF_RE.sub(lambda m: _shift_ref(m, row_offset), formula)


# ----------------------------------------------------------------------
# Read the default row, expanding repeated columns
# ----------------------------------------------------------------------
def read_row_from_zip(zip_path: Path, sheet_name: str, logical_row_idx: int, column_count: int):
    """Return a list of values for the requested logical row, expanding repeated cells."""
    ns = {
        "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
        "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
        "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    }

    table_name_attr = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}name"
    rows_repeated_attr = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}number-rows-repeated"
    cols_repeated_attr = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}number-columns-repeated"
    value_attr = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}value"
    string_value_attr = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}string-value"

    with ZipFile(zip_path) as zf:
        with zf.open("content.xml") as stream:
            context = etree.iterparse(stream, events=("start", "end"))
            in_target_table = False
            logical_row = 0
            for event, elem in context:
                tag = etree.QName(elem).localname
                if event == "start":
                    if tag == "table" and elem.get(table_name_attr) == sheet_name:
                        in_target_table = True
                        logical_row = 0
                elif event == "end":
                    if in_target_table and tag == "table-row":
                        repeat = int(elem.get(rows_repeated_attr, "1"))
                        if logical_row + repeat > logical_row_idx:
                            defaults = []
                            for cell in elem:
                                cell_tag = etree.QName(cell).localname
                                if cell_tag not in {"table-cell", "covered-table-cell"}:
                                    continue

                                repeat_cols = int(cell.get(cols_repeated_attr, "1"))
                                if cell_tag == "covered-table-cell":
                                    value = None
                                else:
                                    paragraphs = []
                                    for p in cell.findall(f"{{{ns['text']}}}p"):
                                        text = "".join(p.itertext()).strip()
                                        if text:
                                            paragraphs.append(text)
                                    value = (
                                        cell.get(value_attr)
                                        or cell.get(string_value_attr)
                                        or (paragraphs[0] if paragraphs else None)
                                    )
                                    if isinstance(value, str):
                                        value = value.strip()
                                        if value == "":
                                            value = None
                                defaults.extend([value] * repeat_cols)

                            if len(defaults) > column_count:
                                defaults = defaults[:column_count]
                            while len(defaults) < column_count:
                                defaults.append(None)
                            return defaults
                        logical_row += repeat
                    if tag == "table" and in_target_table:
                        in_target_table = False

    raise ValueError(f"Could not locate row {logical_row_idx} in sheet '{sheet_name}'")


# ----------------------------------------------------------------------
# Snapshot & restore helpers for protected header rows
# ----------------------------------------------------------------------
def snapshot_rows(sheet, row_indices, raw_values=None):
    """Capture value and formula for the specified rows."""
    snapshots = []
    ncols = sheet.ncols()
    for row in row_indices:
        if row >= sheet.nrows():
            snapshots.append(None)
            continue
        row_snapshot = []
        override = None
        if raw_values and row in raw_values:
            override = raw_values[row]
        for col in range(ncols):
            cell = sheet[row, col]
            if override is not None and col < len(override):
                value = override[col]
            else:
                value = cell.value
            row_snapshot.append((value, cell.formula))
        snapshots.append(row_snapshot)
    return snapshots


TABLE_FORMULA_ATTR = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}formula"


def restore_rows(sheet, row_indices, snapshots):
    """Restore previously captured rows (values and formulas)."""
    ncols = sheet.ncols()
    for idx, row in enumerate(row_indices):
        if row >= sheet.nrows():
            continue
        row_snapshot = snapshots[idx]
        if row_snapshot is None:
            continue
        for col in range(min(ncols, len(row_snapshot))):
            value, formula = row_snapshot[col]
            cell = sheet[row, col]
            if formula:
                cell.formula = formula
                if value is not None:
                    cell.set_value(value)
            else:
                if cell.formula:
                    cell.xmlnode.attrib.pop(TABLE_FORMULA_ATTR, None)
                if value is None:
                    cell.clear()
                else:
                    cell.set_value(value)


# ----------------------------------------------------------------------
# Main autofill routine
# ----------------------------------------------------------------------
def autofill_ods(in_path: Path, out_path: Path | None = None):
    print(f"Opening: {in_path}")
    doc = OdsDocument.load(str(in_path))
    sheet = doc.sheets[0]

    # ------------------------------------------------------------------
    # 1. Locate the “default” row (column A == "default")
    # ------------------------------------------------------------------
    default_row_idx = None
    for r in range(sheet.nrows()):
        if sheet[r, 0].value and str(sheet[r, 0].value).strip().lower() == "default":
            default_row_idx = r
            break
    if default_row_idx is None:
        print("No row with 'default' in column A – nothing to do.")
        return

    print(f"Default row found at index {default_row_idx}")

    # ------------------------------------------------------------------
    # 2. Load defaults (one per column)
    # ------------------------------------------------------------------
    defaults = read_row_from_zip(
        in_path,
        sheet.name,
        default_row_idx,
        sheet.ncols(),
    )
    print(f"Loaded {len(defaults)} default values (sheet reports {sheet.ncols()} columns)")

    # ------------------------------------------------------------------
    # 3. Process each column that has a header name
    # ------------------------------------------------------------------
    start_row = default_row_idx + 1 # first row *below* the default row
    protected_rows = 12 # never touch rows 0-11
    if start_row < protected_rows:
        print(f"Warning: default row ({default_row_idx}) is inside the protected header block.")
        start_row = protected_rows

    protected_indices = list(range(min(protected_rows, sheet.nrows())))
    raw_header_rows = {}
    for row in protected_indices:
        try:
            raw_header_rows[row] = read_row_from_zip(
                in_path,
                sheet.name,
                row,
                sheet.ncols(),
            )
        except ValueError:
            raw_header_rows[row] = None
    protected_snapshot = snapshot_rows(sheet, protected_indices, raw_header_rows)

    for col in range(sheet.ncols()):
        header = sheet[0, col].value
        if not header: # empty header → skip column
            continue

        raw_default = defaults[col]
        if raw_default is None:
            default = None
        else:
            default = str(raw_default).strip()
            if default == "":
                default = None

        if default is None or default.upper() == "NA":
            continue

        # --------------------------------------------------------------
        # Find the first *non-empty* cell **below** the default row
        # --------------------------------------------------------------
        first_val_row = None
        first_is_formula = False
        first_value = None
        first_formula = None
        first_literal = None

        for r in range(start_row, sheet.nrows()):
            cell = sheet[r, col]
            if cell.formula:
                first_is_formula = True
                first_value = cell.value
                first_formula = cell.formula
                first_val_row = r
                break
            if cell.value not in (None, "", " "):
                first_literal = cell.value
                first_val_row = r
                break

        if first_val_row is None:
            continue # nothing to copy from

        # --------------------------------------------------------------
        # Numeric constant
        # --------------------------------------------------------------
        if isinstance(raw_default, (int, float)) or (isinstance(default, str) and default.replace(".", "", 1).lstrip("-").isdigit()):
            num = float(default)
            for r in range(start_row, first_val_row):
                sheet[r, col].set_value(num)
            continue

        # --------------------------------------------------------------
        # Literal string (not "last")
        # --------------------------------------------------------------
        if default != "last":
            for r in range(start_row, first_val_row):
                sheet[r, col].set_value(default)
            continue

        # --------------------------------------------------------------
        # "last" – copy the first non-empty value/formula upwards
        # --------------------------------------------------------------
        if first_is_formula:
            base_formula = first_formula or ""
            if base_formula.startswith("of:="):
                base_formula = base_formula[4:]

            for r in range(start_row, first_val_row):
                offset = r - first_val_row
                shifted = adjust_formula(base_formula, offset)
                cell = sheet[r, col]
                cell.formula = f"of:={shifted}"
                if first_value not in (None, "", " "):
                    cell.set_value(first_value)
            continue

        # Plain value
        for r in range(start_row, first_val_row):
            sheet[r, col].set_value(first_literal)

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    restore_rows(sheet, protected_indices, protected_snapshot)

    if out_path is None:
        out_path = in_path.stem + "_autofilled.ods"
    else:
        out_path = Path(out_path)

    print(f"Saving to: {out_path}")
    doc.save(str(out_path))
    print("Done!")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Autofill ODS columns based on a 'default' row.")
    parser.add_argument("inputs", nargs="+", help="Input .ods file(s).")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file name (only allowed with a single input).",
    )
    args = parser.parse_args()

    if args.output and len(args.inputs) > 1:
        print("Error: --output can only be used with one input file.", file=sys.stderr)
        sys.exit(1)

    for inp in args.inputs:
        in_path = Path(inp)
        out_path = Path(args.output) if args.output else None
        autofill_ods(in_path, out_path)


if __name__ == "__main__":
    main()
