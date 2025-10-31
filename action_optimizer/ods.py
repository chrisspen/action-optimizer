"""Lightweight helpers around odfpy for working with ODS spreadsheets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections.abc import Iterable

from odf import teletype
from odf.namespaces import OFFICENS, TABLENS, TEXTNS
from odf.opendocument import OpenDocument, OpenDocumentSpreadsheet, load
from odf.table import CoveredTableCell, Table, TableCell, TableRow
from odf.text import P


def _is_tag(node, namespace: str, local: str) -> bool:
    """Return True if the node matches the requested namespace/local name."""
    return getattr(node, "qname", None) == (namespace, local)


def _set_element_text(element, text: str | None) -> None:
    """Replace the textual content of *element* with *text*."""
    while element.hasChildNodes():
        element.removeChild(element.firstChild)
    if text:
        paragraph = P()
        paragraph.addText(text)
        element.addElement(paragraph)


def _clone_cell(cell) -> TableCell | CoveredTableCell:
    if _is_tag(cell, TABLENS, "covered-table-cell"):
        clone = CoveredTableCell()
    else:
        clone = TableCell()
    for (ns, name), value in cell.attributes.items():
        clone.setAttrNS(ns, name, value)
    if _is_tag(cell, TABLENS, "table-cell"):
        child = cell.firstChild
        while child is not None:
            next_child = child.nextSibling
            if _is_tag(child, TEXTNS, "p"):
                text = teletype.extractText(child)
                paragraph = P()
                if text:
                    paragraph.addText(text)
                clone.addElement(paragraph)
            child = next_child
    return clone


def _clone_row(row: TableRow) -> TableRow:
    clone = TableRow()
    for (ns, name), value in row.attributes.items():
        if ns == TABLENS and name == "number-rows-repeated":
            continue
        clone.setAttrNS(ns, name, value)
    child = row.firstChild
    while child is not None:
        next_child = child.nextSibling
        if _is_tag(child, TABLENS, "table-cell") or _is_tag(child, TABLENS, "covered-table-cell"):
            clone.addElement(_clone_cell(child))
        child = next_child
    return clone


def _expand_repeated_cells(row: TableRow) -> list[TableCell | CoveredTableCell]:
    """Expand `table:number-columns-repeated` attributes into explicit cells."""
    cells: list[TableCell | CoveredTableCell] = []
    node = row.firstChild
    while node is not None:
        next_node = node.nextSibling
        if _is_tag(node, TABLENS, "table-cell") or _is_tag(node, TABLENS, "covered-table-cell"):
            repeat_attr = node.getAttrNS(TABLENS, "number-columns-repeated")
            repeat = int(repeat_attr) if repeat_attr else 1
            if repeat > 1 and repeat_attr is not None:
                node.removeAttrNS(TABLENS, "number-columns-repeated")
            cells.append(node)
            insert_after = node
            for _ in range(repeat - 1):
                clone = _clone_cell(node)
                row.insertBefore(clone, insert_after.nextSibling)
                insert_after = clone
                cells.append(clone)
        node = next_node
    return cells


@dataclass
class RowBlock:
    element: TableRow
    repeat: int
    expanded: bool


@dataclass
class OdsCell:
    """Wrapper providing convenient access to a table cell."""

    sheet: OdsSheet
    row_element: TableRow
    col_index: int

    def _element(self) -> TableCell | CoveredTableCell:
        cells = self.sheet._get_row_cells(self.row_element)
        return cells[self.col_index]

    def _ensure_table_cell(self) -> TableCell:
        element = self._element()
        if _is_tag(element, TABLENS, "table-cell"):
            return element
        new_cell = TableCell()
        row = self.row_element
        row.insertBefore(new_cell, element)
        row.removeChild(element)
        row_cells = self.sheet._get_row_cells(row)
        row_cells[self.col_index] = new_cell
        return new_cell

    @property
    def formula(self) -> str | None:
        element = self._element()
        formula = element.getAttrNS(TABLENS, "formula")
        return formula or None

    @formula.setter
    def formula(self, value: str | None) -> None:
        element = self._ensure_table_cell()
        if value:
            element.setAttrNS(TABLENS, "formula", value)
        else:
            if element.getAttrNS(TABLENS, "formula") is not None:
                element.removeAttrNS(TABLENS, "formula")

    @property
    def value(self):
        element = self._element()
        value_type = element.getAttrNS(OFFICENS, "value-type")
        if value_type == "float":
            value_attr = element.getAttrNS(OFFICENS, "value")
            if value_attr:
                try:
                    return float(value_attr)
                except ValueError:
                    return value_attr
        string_value = element.getAttrNS(OFFICENS, "string-value")
        if string_value:
            return string_value
        text_value = teletype.extractText(element).strip()
        return text_value or None

    def set_value(self, value) -> None:
        element = self._ensure_table_cell()
        if value is None:
            self.clear()
            return

        if isinstance(value, (int, float)):
            element.setAttrNS(OFFICENS, "value-type", "float")
            element.setAttrNS(OFFICENS, "value", str(value))
            if element.getAttrNS(OFFICENS, "string-value") is not None:
                element.removeAttrNS(OFFICENS, "string-value")
            _set_element_text(element, str(value))
            return

        element.setAttrNS(OFFICENS, "value-type", "string")
        element.setAttrNS(OFFICENS, "string-value", str(value))
        if element.getAttrNS(OFFICENS, "value") is not None:
            element.removeAttrNS(OFFICENS, "value")
        _set_element_text(element, str(value))

    def clear(self) -> None:
        element = self._ensure_table_cell()
        for local in ("value", "value-type", "string-value", "boolean-value", "date-value", "time-value"):
            if element.getAttrNS(OFFICENS, local) is not None:
                element.removeAttrNS(OFFICENS, local)
        _set_element_text(element, None)

    @property
    def xmlnode(self):
        return self._element()


class OdsSheet:
    """Abstraction over an ODS sheet."""

    def __init__(self, table: Table):
        self._table = table
        self.name = table.getAttrNS(TABLENS, "name") or ""
        self._row_blocks: list[RowBlock] = []
        self._row_cells: dict[int, list[TableCell | CoveredTableCell]] = {}
        self._ncols = 0

        node = table.firstChild
        while node is not None:
            next_node = node.nextSibling
            if _is_tag(node, TABLENS, "table-row"):
                repeat_attr = node.getAttrNS(TABLENS, "number-rows-repeated")
                repeat = int(repeat_attr) if repeat_attr else 1
                expanded = repeat == 1
                self._row_blocks.append(RowBlock(node, repeat, expanded))
            node = next_node

    def _locate_block(self, index: int) -> tuple[int, RowBlock, int]:
        total = 0
        for idx, block in enumerate(self._row_blocks):
            if index < total + block.repeat:
                offset = index - total
                return idx, block, offset
            total += block.repeat
        raise IndexError("Row index out of bounds")

    def _expand_row_block(self, block_idx: int, offset: int) -> TableRow:
        block = self._row_blocks[block_idx]
        element = block.element
        repeat = block.repeat
        consumed = offset + 1

        if element.getAttrNS(TABLENS, "number-rows-repeated") is not None:
            element.removeAttrNS(TABLENS, "number-rows-repeated")

        new_blocks: list[RowBlock] = [RowBlock(element, 1, True)]
        previous = element

        for _ in range(1, consumed):
            clone = _clone_row(element)
            self._table.insertBefore(clone, previous.nextSibling)
            previous = clone
            new_blocks.append(RowBlock(clone, 1, True))

        remaining = repeat - consumed
        if remaining > 0:
            remainder = _clone_row(element)
            remainder.setAttrNS(TABLENS, "number-rows-repeated", str(remaining))
            self._table.insertBefore(remainder, previous.nextSibling)
            new_blocks.append(RowBlock(remainder, remaining, False))

        self._row_blocks[block_idx:block_idx + 1] = new_blocks
        return new_blocks[offset].element

    def _ensure_row(self, index: int) -> TableRow:
        if index < 0:
            raise IndexError("Row index out of bounds")
        block_idx, block, offset = self._locate_block(index)
        if block.repeat == 1:
            block.expanded = True
            return block.element
        return self._expand_row_block(block_idx, offset)

    def _get_row_cells(self, row: TableRow) -> list[TableCell | CoveredTableCell]:
        key = id(row)
        cells = self._row_cells.get(key)
        if cells is None:
            cells = _expand_repeated_cells(row)
            self._row_cells[key] = cells
            self._ncols = max(self._ncols, len(cells))
        return cells

    def _ensure_column(self, row: TableRow, col_idx: int) -> None:
        if col_idx < 0:
            raise IndexError("Column index out of bounds")
        cells = self._get_row_cells(row)
        while col_idx >= len(cells):
            new_cell = TableCell()
            row.addElement(new_cell)
            cells.append(new_cell)
        self._ncols = max(self._ncols, col_idx + 1)

    def nrows(self) -> int:
        return sum(block.repeat for block in self._row_blocks)

    def ncols(self) -> int:
        if self._ncols == 0 and self._row_blocks:
            self._get_row_cells(self._ensure_row(0))
        return self._ncols

    def __getitem__(self, key: tuple[int, int]) -> OdsCell:
        row_idx, col_idx = key
        row = self._ensure_row(row_idx)
        self._ensure_column(row, col_idx)
        return OdsCell(self, row, col_idx)


class OdsDocument:
    """Simple facade around odfpy's document object."""

    def __init__(self, document: OpenDocument):
        self._document = document
        tables: Iterable[Table] = document.spreadsheet.getElementsByType(Table)
        self.sheets: list[OdsSheet] = [OdsSheet(table) for table in tables]

    @classmethod
    def load(cls, path: str | bytes) -> OdsDocument:
        document = load(path)
        return cls(document)

    @classmethod
    def new_blank(cls, sheet_name: str, rows: int, cols: int) -> OdsDocument:
        document = OpenDocumentSpreadsheet()
        table = Table(name=sheet_name)
        for _ in range(rows):
            row = TableRow()
            for _ in range(cols):
                row.addElement(TableCell())
            table.addElement(row)
        document.spreadsheet.addElement(table)
        return cls(document)

    def save(self, path: str) -> None:
        self._document.save(path)

    @property
    def raw_document(self) -> OpenDocument:
        return self._document
