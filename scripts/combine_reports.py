#!/usr/bin/env python
import os
import sys
from datetime import date
from collections import OrderedDict
import pathlib

from pyexcel_ods import save_data
from pandas_ods_reader import read_ods
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.worksheet.dimensions import ColumnDimension

# pylint: disable=wrong-import-position

sys.path.insert(1, '..')
from action_optimizer.optimizer import Optimizer

current_path = pathlib.Path(__file__).parent.resolve()

dt = date.today()
if len(sys.argv) > 1:
    dt = sys.argv[1]


def auto_size_and_fix_columns(input_ods, postfix=None):
    # Generate the output_xlsx filename
    postfix = postfix or ''
    output_xlsx = os.path.splitext(input_ods)[0] + postfix + '.xlsx'

    # Load the existing ODS file as a DataFrame
    df = pd.read_excel(input_ods, engine='odf')

    # Save the DataFrame to an Excel file (since openpyxl works with Excel files)
    df.to_excel(output_xlsx, index=False)

    # Load the Excel file with openpyxl
    wb = load_workbook(output_xlsx)
    ws = wb.active

    # Apply bold formatting to the header row
    for cell in ws[1]:
        cell.font = Font(bold=True)

    # Auto-size the columns
    for col in ws.columns:
        max_length = 0
        column = col[0].column_letter
        for cell in col:
            max_length = max(max_length, len(str(cell.value)))
        adjusted_width = (max_length + 2)
        ws.column_dimensions[column].width = adjusted_width

    # Save the changes to the Excel file
    wb.save(output_xlsx)


def combine():

    datafile = os.environ.get('ACTION_OPTIMIZER_DATAFILE')
    print(f'Loading data file: {datafile}')
    optimizer = Optimizer(datafile)
    last_data = optimizer.get_first_nonblank_row(headers=True)

    pcc_path = os.path.abspath(f'{current_path}/../reports/{dt}/pcc.ods')
    print(f'Using PCC path: {pcc_path}')
    assert os.path.isfile(pcc_path)

    ens_path = os.path.abspath(f'{current_path}/../reports/{dt}/ensemble.ods')
    print(f'Using ENS path: {ens_path}')
    assert os.path.isfile(ens_path)

    pcc_df = read_ods(pcc_path, 1, headers=True)
    ens_df = read_ods(ens_path, 1, headers=True)

    feature_values = {} # {name: [values]}

    pcc_confidences = {}
    ens_confidences = {}

    pcc_values = {}
    ens_values = {}

    for row in pcc_df.to_dict(orient='records'):
        name = row['name']
        confidence = row['pcc_zero']
        best_value = row['best_value']
        feature_values.setdefault(name, [])
        feature_values[name].append(confidence)
        pcc_values[name] = best_value
        pcc_confidences[name] = confidence

    for row in ens_df.to_dict(orient='records'):
        name = row['name']
        confidence = row['confidence']
        best_value = row['best value']
        if not best_value:
            confidence = 1 - confidence
        feature_values.setdefault(name, [])
        feature_values[name].append(confidence)
        ens_values[name] = best_value
        ens_confidences[name] = confidence

    final_values = [] # [(score, name)]
    for name, confidences in feature_values.items():
        mean_confidence = sum(confidences) / len(confidences)
        final_values.append((mean_confidence, name))
    final_values.sort()
    for name, conf in final_values:
        print(name, conf)

    combine_fn = os.path.abspath(f'{current_path}/../reports/{dt}/combine-{dt}.ods')
    print(f'Saving {combine_fn}.')
    data = OrderedDict()
    rows = []
    for i, (conf, name) in enumerate(final_values, 2):
        pcc_conf = pcc_confidences.get(name, 0.5)
        ens_conf = ens_confidences.get(name, 0.5)
        algo_conf = (pcc_conf + ens_conf) / 2
        rec_change = 0
        if name in last_data:
            rec_change = bool(last_data[name]) != bool(round(algo_conf))
        rows.append([name, conf, pcc_conf, ens_conf, 0.5, f'=(C{i}+D{i}+E{i})/3', pcc_values.get(name, ''), ens_values.get(name, ''), rec_change])
    data.update({"Sheet 1": [
        ['name', 'score', 'pcc_conf', 'ens_conf', 'human_conf', 'agg_conf', 'pcc_value', 'ens_value', 'recommend_change']] + \
        rows
                 })
    save_data(combine_fn, data)
    print(f'Saved {combine_fn}.')

    auto_size_and_fix_columns(combine_fn, '-final')


if __name__ == "__main__":
    combine()
