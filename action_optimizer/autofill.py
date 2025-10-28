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

import ezodf
from lxml import etree

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
def read_defaults_row(sheet, default_row_idx):
    """Return a list of default values, one per column, expanding repeated cells."""
    ns = {
        "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
        "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
        "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    }

    row_node = sheet.xmlnode.find(f".//{{*}}table-row[{default_row_idx + 1}]", namespaces=ns) # xpath is 1-based

    defaults = []
    for cell in row_node.iterfind("./{*}table-cell", namespaces=ns):
        repeat = int(cell.get(
            "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}number-columns-repeated",
            "1",
        ))
        # value can be in office:value, office:string-value or text:p
        value = (
            cell.get("{urn:oasis:names:tc:opendocument:xmlns:office:1.0}value") or cell.get("{urn:oasis:names:tc:opendocument:xmlns:office:1.0}string-value")
            or next((p.text for p in cell.iterfind("./{*}p", namespaces=ns)), None)
        )
        if value is not None:
            value = value.strip() or None
        defaults.extend([value] * repeat)

    # Pad to the real column count (ezodf may report more columns than stored)
    while len(defaults) < sheet.ncols():
        defaults.append(None)

    return defaults


# ----------------------------------------------------------------------
# Main autofill routine
# ----------------------------------------------------------------------
def autofill_ods(in_path: Path, out_path: Path | None = None):
    print(f"Opening: {in_path}")
    doc = ezodf.opendoc(str(in_path))
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
    defaults = read_defaults_row(sheet, default_row_idx)
    print(f"Loaded {len(defaults)} default values (sheet reports {sheet.ncols()} columns)")

    # ------------------------------------------------------------------
    # 3. Process each column that has a header name
    # ------------------------------------------------------------------
    start_row = default_row_idx + 1 # first row *below* the default row
    protected_rows = 12 # never touch rows 0-11
    if start_row < protected_rows:
        print(f"Warning: default row ({default_row_idx}) is inside the protected header block.")
        start_row = protected_rows

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
        first_raw = None

        for r in range(start_row, sheet.nrows()):
            cell = sheet[r, col]
            if cell.formula:
                first_raw = cell.formula
                first_is_formula = True
                first_val_row = r
                break
            if cell.value not in (None, "", " "):
                first_raw = cell.value
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
            # ODS stores formulas as "of:=" prefix
            base_formula = first_raw
            if base_formula.startswith("of:="):
                base_formula = base_formula[4:]

            for r in range(start_row, first_val_row):
                offset = r - first_val_row
                shifted = adjust_formula(base_formula, offset)
                sheet[r, col].formula = f"of:={shifted}"
        else:
            # plain value
            for r in range(start_row, first_val_row):
                sheet[r, col].set_value(first_raw)

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    if out_path is None:
        out_path = in_path.stem + "_autofilled.ods"
    else:
        out_path = Path(out_path)

    print(f"Saving to: {out_path}")
    doc.saveas(str(out_path))
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
