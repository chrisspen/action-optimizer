#!/usr/bin/env python
import os
from datetime import date
from collections import OrderedDict
import pathlib

from pyexcel_ods import save_data
from pandas_ods_reader import read_ods

current_path = pathlib.Path(__file__).parent.resolve()

pcc_path = os.path.abspath(f'{current_path}/../reports/{date.today()}/pcc.ods')
print(f'Using PCC path: {pcc_path}')
assert os.path.isfile(pcc_path)

ens_path = os.path.abspath(f'{current_path}/../reports/{date.today()}/ensemble.ods')
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

combine_fn = os.path.abspath(f'{current_path}/../reports/{date.today()}/combine-{date.today()}.ods')
print(f'Saving {combine_fn}.')
data = OrderedDict()
data.update({"Sheet 1": [
    ['name', 'score', 'pcc_conf', 'ens_conf', 'human_conf', 'agg_conf', 'pcc_value', 'ens_value']] + \
    [
        [name, conf, pcc_confidences.get(name, 0.5), ens_confidences.get(name, 0.5), 0.5,  f'=(C{i}+D{i}+E{i})/3',
        pcc_values.get(name, ''), ens_values.get(name, '')] for i, (conf, name) in enumerate(final_values, 2)]
             })
save_data(combine_fn, data)
print(f'Saved {combine_fn}.')
