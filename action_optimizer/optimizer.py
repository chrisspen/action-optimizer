#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import traceback
import copy
import csv
import random
import math
import logging
from datetime import date, timedelta
from pprint import pprint
from decimal import Decimal
from collections import defaultdict

import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.stats import norm, binned_statistic
from dateutil.parser import parse
from pyexcel_ods import get_data
from weka.arff import ArffFile, Nom, Num, Str, Date, MISSING
from weka.classifiers import EnsembleClassifier

from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s", "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)


class SkipRow(Exception):
    pass


BASE_DIR = os.path.split(os.path.realpath(__file__))[0]

NUMERIC = 'numeric'
DATE = 'date'

MN = 'mn_' # morning
NO = 'no_' # noon
EV = 'ev_' # evening
MONTHLY = '_monthly'
OTHERS = 'others'
# MANAGE = 'manage'

NA_CHANGE = ''
RELATIVE_CHANGE = 'rel'
ABSOLUTE_CHANGE = 'abs'
CHANGE_TYPES = (NA_CHANGE, RELATIVE_CHANGE, ABSOLUTE_CHANGE)

# The minimum number of days of data needed before a prediction can be made.
MIN_DAYS = 30

HEADER_ROW_INDEX = 0
TYPE_ROW_INDEX = 1
RANGE_ROW_INDEX = 2
# CONTROLLABLE_ROW_INDEX = 3
LEARN_ROW_INDEX = 3 # These columns are given to the classifiers to learn.
PREDICT_ROW_INDEX = 4 # These columns are individually looked at to predict an optimal change.
CHANGE_ROW_INDEX = 5 # 0=none, 1=relative change recommendation, 2=absolute value recommendation
DATA_ROW_INDEX = 6
TOLERANCE = 1.0
#CLASS_ATTR_NAME = 'score'
#CLASS_ATTR_NAME = 'score_change'
CLASS_ATTR_NAME = 'score_next'
DEFAULT_SCORE_FIELD_NAME = 'score'
DEFAULT_CLASSIFIER_FN = '/tmp/%s-last-classifier.pkl.gz'
DEFAULT_RELATION = '%s-training'

BASE_REPORTS_DIR = './reports'


def attempt_cast_str_to_numeric(value):
    try:
        return float(value)
    except ValueError:
        return value


def linear_func(x, m=1, b=0):
    """
    m = slope
    b = y-intercept
    """
    y = m * x + b
    return y


def sigmoid_func(x, x0=0, k=1, a=1, c=0):
    """
    Generates an S-surve.

    y = 1/(1 + np.exp(-x)) # low left to high right

    x0 = horizontal offset
    k = polarity?
    a = magnitude?
    c = vertical offset
    """
    if isinstance(x, (tuple, list)):
        x = np.array(x)
    y = a / (1 + np.exp(-k * (x - x0))) + c
    return y


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def gaussian_func(x, a=1, b=0, c=1, d=0):
    """
    Generates a bell curve.

    a = magnitude
    b = mu/average/mean
    c = sigma/variance
    d = vertical offset
    """
    return a * exp(-(x - b)**2 / (2 * c**2)) + d


def fit_linear(x, y):
    popt, pcov = scipy.optimize.curve_fit(linear_func, x, y)
    return linear_func(x, *popt)


def fit_sigmoid(x, y):
    p0 = [max(y), np.median(x), 1, min(y)]
    popt, pcov = scipy.optimize.curve_fit(sigmoid_func, x, y, p0, method='dogbox')
    return sigmoid_func(x, *popt)


def fit_gaussian(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
    popt, pcov = scipy.optimize.curve_fit(gaussian_func, x, y, p0=[max(y), mean, sigma, 0])
    return gaussian_func(x, *popt)


class Optimizer:

    def __init__(self, fn, **kwargs):

        if not fn.startswith('/'):
            fn = os.path.join(BASE_DIR, fn)
        assert os.path.isfile(fn), f'File {fn} does not exist.'
        self.fn = fn

        # Find the fully qualified absolute file path, minus the extension.
        self.fqfn_base = os.path.splitext(os.path.abspath(fn))[0]

        # Find the filename, minus any path or extension.
        self.fn_base = os.path.splitext(os.path.split(fn)[-1])[0]

        self.__dict__.update(kwargs)
        self.score_field_name = self.__dict__.get('score_field_name') or DEFAULT_SCORE_FIELD_NAME
        self.only_attribute = self.__dict__.get('only_attribute')
        self.stop_on_error = self.__dict__.get('stop_on_error') or False
        self.no_train = self.__dict__.get('no_train') or False
        self.all_classifiers = self.__dict__.get('all_classifiers') or False
        self.calculate_pcc = self.__dict__.get('calculate_pcc') or False
        self.plot_attributes = [_.strip() for _ in (self.__dict__.get('plot_attributes') or '').strip().split(',') if _.strip()]
        self.yes = self.__dict__.get('yes', None)
        self.classifier_fn = self.__dict__.get('classifier_fn', DEFAULT_CLASSIFIER_FN) % self.fn_base
        self.relation = self.__dict__.get('relation', DEFAULT_RELATION) % self.fn_base

    def plot(self, x, y, name, show=False):

        logger.info('Plotting functional estimates for %s.', name)
        plt.clf()

        # Generate pure sigmoid curve.
        plt.scatter(x, y, label='Raw', color='red', marker='.')

        # Estimate the original curve from the noise.
        linear_estimate = fit_linear(x, y)
        logger.info('Linear estimate: %s', linear_estimate)

        try:
            gaussian_estimate = fit_gaussian(x, y)
            logger.info('Gaussian estimate: %s', gaussian_estimate)
        except RuntimeError as exc:
            logger.warning('Unable to fit gaussian for %s: %s', name, exc)
            gaussian_estimate = None

        try:
            sigmoid_estimate = fit_sigmoid(x, y)
            logger.info('Sigmoid estimate: %s', sigmoid_estimate)
        except RuntimeError as exc:
            logger.warning('Unable to fit sigmoid for %s: %s', name, exc)
            sigmoid_estimate = None

        linear_cod = gaussian_cod = sigmoid_cod = None
        linear_error = gaussian_error = sigmoid_error = None

        try:
            linear_cod = r2_score(y, linear_estimate)
            linear_error = abs(1 - linear_cod)
        except (TypeError, ValueError) as exc:
            logger.warning('Unable to calculate linear error: %s', exc)

        try:
            gaussian_cod = r2_score(y, gaussian_estimate)
            print('gaussian_cod:', gaussian_cod)
            gaussian_error = abs(1 - gaussian_cod)
            print('gaussian_error:', gaussian_error)
        except (TypeError, ValueError) as exc:
            logger.warning('Unable to calculate gaussian error: %s', exc)

        try:
            sigmoid_cod = r2_score(y, sigmoid_estimate)
            sigmoid_error = abs(1 - sigmoid_error)
        except (TypeError, ValueError) as exc:
            logger.warning('Unable to calculate sigmoid error: %s', exc)

        binned_result = binned_statistic(x, y, statistic='median', bins=10)
        binned_x = []
        binned_y = binned_result.statistic
        best_bin = (None, None) # (median, bin center)
        for i in range(len(binned_result.bin_edges) - 1):
            _x = (binned_result.bin_edges[i] + binned_result.bin_edges[i + 1]) / 2.
            _y = binned_y[i]
            binned_x.append(_x)
            _best_value, _best_i = best_bin
            if _best_value is None:
                best_bin = _y, _x
            else:
                best_bin = max(best_bin, (_y, _x))
        best_binned_value, best_binned_center = best_bin

        if linear_estimate is not None:
            plt.scatter(x, linear_estimate, label='Linear', marker='.')
        if gaussian_estimate is not None:
            plt.scatter(x, gaussian_estimate, label='Guassian', marker='.')
        if sigmoid_estimate is not None:
            plt.scatter(x, sigmoid_estimate, label='Sigmoid', marker='.')

        plt.plot(binned_x, binned_y, label='Binned Median')

        plt.title(name)
        plt.legend()
        plt.savefig(f'{BASE_REPORTS_DIR}/{date.today()}/{name}.png')
        if show:
            plt.show()

        return linear_error, gaussian_error, sigmoid_error, best_binned_value, best_binned_center

    def analyze(self, save=True):
        self.score_field_name = self.score_field_name or DEFAULT_SCORE_FIELD_NAME

        def isolate_attr(rows, attr):
            _x = []
            _y = []
            for row in rows:
                try:
                    xv = float(row[attr].value)
                    yv = float(row[CLASS_ATTR_NAME].value)
                    _x.append(xv)
                    _y.append(yv)
                except (KeyError, AttributeError):
                    continue
            x = np.array(_x).astype(np.float32)
            y = np.array(_y).astype(np.float32)
            return x, y

        logger.info('Retrieving data...')
        sys.stdout.flush()
        data = get_data(self.fn)['data']

        # Validate header rows.
        field_to_day_count = {} # {name: number of days of data}
        column_names = data[HEADER_ROW_INDEX]
        column_types = data[TYPE_ROW_INDEX]
        column_types_dict = dict(zip(column_names, column_types))
        column_ranges = dict(zip(column_names, data[RANGE_ROW_INDEX])) # min,max,step
        for _name in column_ranges:
            if _name not in ('date') and column_ranges[_name]:
                column_ranges[_name] = list(map(float, column_ranges[_name].split(',')))
            else:
                column_ranges[_name] = None

        column_nominals = self.column_nominals = {} # {name: set(nominals)}
        assert len(column_names) == len(column_types)
        for column_name, ct in zip(column_names, column_types):
            assert ct and (ct in (DATE, NUMERIC) or (ct[0] == '{' and ct[-1] == '}')), f'Column "{column_name}" has invalid type "{ct}"'
            if ct[0] == '{':
                try:
                    column_nominals[column_name] = set(ct[1:-1].split(','))
                except IndexError as exc:
                    raise Exception(f'Error processing nominal value for column "{column_name}": "{ct}"') from exc

        column_learnables = self.column_learnables = {}
        for _a, _b in zip(column_names, data[LEARN_ROW_INDEX]):
            if _a == DATE:
                column_learnables[_a] = 0
                continue
            try:
                column_learnables[_a] = int(_b)
            except Exception as exc:
                raise Exception(f'Error checking controllable for column {_a}: {exc}') from exc
        logger.info('column_learnables: %s', column_learnables)

        column_predictables = self.column_predictables = {}
        for _a, _b in zip(column_names, data[PREDICT_ROW_INDEX]):
            if _a == DATE:
                column_predictables[_a] = 0
                continue
            try:
                column_predictables[_a] = int(_b)
            except Exception as exc:
                raise Exception(f'Error checking predictable for column {_a}: {exc}') from exc
        logger.info('column_predictables: %s', column_predictables)

        column_changeables = self.column_changeables = {}
        for _a, _b in zip(column_names, data[CHANGE_ROW_INDEX]):
            if _a == DATE:
                column_changeables[_a] = NA_CHANGE
                continue
            try:
                assert _b in CHANGE_TYPES, f'Invalid change type for column {_a}: {_b}'
                column_changeables[_a] = _b
            except Exception as exc:
                raise Exception(f'Error checking changeable for column {_a}: {exc}') from exc
        logger.info('column_changeables: %s', column_changeables)

        # Load data rows and convert to ARFF format.
        row_errors = {} # {row_count: error}
        data = data[DATA_ROW_INDEX:]
        arff = ArffFile(relation=self.relation)
        arff.class_attr_name = CLASS_ATTR_NAME
        arff.relation = self.relation # 'optimizer-training'
        row_count = 0
        best_day = -1e999999999999, None # (score, data)
        best_date = -1e999999999999, None
        last_full_day = date.min, None # (date, data)
        date_to_score = {} # {date: score}
        # date_to_row = {} # {date: row}
        column_values = defaultdict(set)
        new_rows = []
        for row in data:
            row_count += 1
            try:
                if not row:
                    continue
                assert len(row) == len(column_names), f"Row {row_count} has length {len(row)} but there are {len(column_names)} column headers."
                assert len(row) == len(column_types)
                old_row = dict(zip(column_names, row))
                new_row = {}
                for row_value, column_name, ct in zip(row, column_names, column_types):

                    # Ignore impartially filled in row for the current day.
                    if column_name.startswith('next_day') or column_name.startswith('subscore'):
                        if row_value == '':
                            raise SkipRow
                        # Remove next_day_* attributes, since these are only used for calculating the score, not predicting it.
                        continue

                    if ct == DATE:
                        if row_count == 1 and not isinstance(row_value, date):
                            # Ignore invalid date on first row, since we purposefully leave this blank.
                            logging.warning('Warning: Invalid date "%s" on row %s.', row_value, row_count)
                            raise SkipRow
                        if isinstance(row_value, str):
                            # If the cell data wasn't entered correctly, the date value might be stored as a string.
                            _row_value = parse(row_value)
                            if _row_value:
                                row_value = _row_value.date()
                                old_row[column_name] = row_value
                        assert isinstance(row_value, date), f'Invalid date "{row_value}" on row {row_count}.'
                        continue

                    if ct == NUMERIC:
                        if row_value != '':
                            row_value = attempt_cast_str_to_numeric(row_value)
                            assert isinstance(row_value, (int, bool, float)), \
                                f'Invalid numeric value "{row_value}" of type "{type(row_value)}" in column "{column_name}" of row {row_count}.'
                            new_row[column_name] = Num(row_value)
                        else:
                            # Otherwise, ignore empty cell values, which means the data is missing.
                            continue
                    elif ct[0] == '{':
                        if row_value != '':
                            _legal_value_list = ', '.join(sorted(map(str, column_nominals[column_name])))
                            assert str(row_value) in column_nominals[column_name], \
                                f'Invalid nominal value "{row_value}" for column "{column_name}". Legal values: {_legal_value_list}'
                            new_row[column_name] = Nom(str(row_value))
                        else:
                            # Otherwise, ignore empty cell values, which means the data is missing.
                            continue
                    else:
                        raise NotImplementedError(f'Unknown type/column: {ct}/{column_name}')

                    column_values[column_name].add(new_row[column_name])
                    field_to_day_count.setdefault(column_name, 0)
                    field_to_day_count[column_name] += new_row[column_name] != '' and new_row[column_name] is not None

                new_row['date'] = old_row['date']
                assert isinstance(old_row['date'], date)
                date_to_score[old_row['date']] = new_row[self.score_field_name]
                #logging.info("new_row:'%s':value: %s", self.score_field_name, new_row[self.score_field_name].value)
                best_day = max(best_day, (new_row[self.score_field_name].value, new_row), key=lambda o: o[0])
                best_date = max(best_date, (new_row[self.score_field_name].value, old_row['date']), key=lambda o: o[0])
                last_full_day = max(last_full_day, (old_row['date'], new_row), key=lambda o: o[0])
                new_rows.append(new_row)
            except SkipRow:
                pass
            except Exception as exc:
                traceback.print_exc()
                row_errors[row_count] = traceback.format_exc()
                if self.stop_on_error:
                    raise

        # Re-score rows to calculate score change per day.
        # Note, we assume the features on the previous day are what predict the score on the next day,
        # so for each row, we discard its score and replace it with the score for the next day.
        # If not next day score is available, we skip that row entirely.
        assert new_rows, "No data!"
        modified_rows = [] # Rows containing a differential score.
        for new_row in new_rows:
            current_date = new_row['date']
            current_score = new_row[self.score_field_name]
            next_date = current_date + timedelta(days=1)
            assert isinstance(next_date, date)
            if next_date in date_to_score:
                next_score = date_to_score[next_date]
                logger.info('current_date: %s', current_date)
                logger.info('current_score.value: %s', current_score.value)
                logger.info('next_date: %s', next_date)
                logger.info('next_score.value: %s', next_score.value)
                assert current_date < next_date

                # CS 2018.9.26 Disabled because I think this may be stalling, when it doesn't see an incremental improvement.
                # Reverting to total score prediction.
                #
                # The prediction target is the change in day to day score, with the intent being to maximize this increase.
                # score_change = Num(next_score.value - current_score.value)
                # logger.info('score_change:', score_change)
                # new_row[CLASS_ATTR_NAME] = score_change

                # The prediction target is the next day's score, with the intent being to maximize this score.
                new_row[CLASS_ATTR_NAME] = next_score

                # Exclude columns from the dataset that are marked as not being fields that should be fed to the learning algorithms.
                for _column, _controllable in column_learnables.items():
                    if not _controllable and _column in new_row:
                        del new_row[_column]

                logger.info('new_row: %s', new_row)
                sys.stdout.flush()
                modified_rows.append(new_row)
                arff.append(new_row)

        if self.plot_attributes:
            for attr_name in self.plot_attributes:
                assert attr_name in column_types_dict, f'Unknown attribute: {attr_name}'
                assert column_types_dict[attr_name] == NUMERIC, f'Attribute {attr_name} is not numeric.'
                x, y = isolate_attr(modified_rows, attr_name)
                self.plot(x, y, attr_name, show=True)
            return

        if self.calculate_pcc:
            # https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
            pcc_rows = []
            report_dir = os.path.join(BASE_REPORTS_DIR, str(date.today()))
            os.makedirs(report_dir, exist_ok=True)
            with open(os.path.join(report_dir, 'pcc.csv'), 'w', encoding='ascii') as fout:
                fieldnames = [
                    'name',
                    'samples',
                    'pcc',
                    'utility',
                    'linear_error',
                    'gaussian_error',
                    'sigmoid_error',
                    'best_binned_value',
                    'best_binned_center',
                ]
                writer = csv.DictWriter(fout, fieldnames=fieldnames)
                writer.writerow(dict(zip(fieldnames, fieldnames)))
                for target_attr in column_names:
                    if column_types_dict[target_attr] != NUMERIC or not column_predictables.get(target_attr) or target_attr == CLASS_ATTR_NAME:
                        continue
                    x, y = isolate_attr(modified_rows, target_attr)

                    pcc = np.corrcoef(y, x)[0, 1]
                    logger.info('Pearson correlation for %s: %s', target_attr, pcc)
                    samples = len(x)
                    if math.isnan(pcc):
                        logger.warning('Skipping attribute "%s" because PCC could not be calculated.', target_attr)
                        continue

                    linear_error, gaussian_error, sigmoid_error, best_binned_value, best_binned_center = self.plot(x, y, target_attr, show=False)
                    print('linear_error, gaussian_error, sigmoid_error:', linear_error, gaussian_error, sigmoid_error)
                    assert gaussian_error is not None

                    pcc_rows.append(
                        dict(
                            name=target_attr,
                            pcc=pcc,
                            samples=samples,
                            utility=samples * pcc,
                            linear_error=linear_error,
                            gaussian_error=gaussian_error,
                            sigmoid_error=sigmoid_error,
                            best_binned_value=best_binned_value,
                            best_binned_center=best_binned_center,
                        )
                    )
                pcc_rows.sort(key=lambda o: o['utility'])
                for pcc_row in pcc_rows:
                    writer.writerow(pcc_row)
            return

        # Cleanup training arff.
        logger.info('attributes: %s', sorted(arff.attributes))
        arff.alphabetize_attributes()
        assert len(arff), 'Empty arff!' # pylint: disable=len-as-condition

        # Report any processing errors on each row.
        if row_errors:
            logger.info('=' * 80)
            logger.info('Row Errors: %s', len(row_errors))
            for row_count in sorted(row_errors):
                logger.info('Row %i:', row_count)
                logger.info(row_errors[row_count])
            logger.info('=' * 80)
        else:
            logger.info('No row errors.')

        # Ensure the base arff file has all nominals values, even if they weren't used.
        for _name, _value in column_nominals.items():
            if _name in arff.attribute_data:
                arff.attribute_data[_name].update(_value)

        training_fn = os.path.join(BASE_DIR, self.fqfn_base + '.arff')
        logger.info('training_fn: %s', training_fn)

        logger.info('Writing arff...')
        with open(training_fn, 'w', encoding='ascii') as fout:
            arff.write(fout)
        logger.info('Arff written!')

        # Train all Weka regressors on arff training file.
        if self.all_classifiers:
            classes = None
        else:
            # See http://bio.med.ucm.es/docs/weka/weka/classifiers/Classifier.html for a complete list.
            classes = [
                # IBk takes ~20 seconds with a CE of 1.
                'weka.classifiers.lazy.IBk',
                # KStar takes ~2000 seconds with a CE of 1.
                'weka.classifiers.lazy.KStar',
                # 2021.3.14 Disabled. Takes forever, usually wrong, and started returning a correlation_coefficient of None.
                # MultilayerPerceptron takes ~5000 seconds.
                # 'weka.classifiers.functions.MultilayerPerceptron',
                'weka.classifiers.functions.LinearRegression',
            ]
        if self.no_train:
            assert os.path.isfile(self.classifier_fn), \
                f'If training is disabled, then a classifier file must exist to re-use, but {self.classifier_fn} does not exist.'
            logger.info('Loading classifier from file %s...', self.classifier_fn)
            classifier = EnsembleClassifier.load(self.classifier_fn)
            logger.info('Classifier loaded.')
        else:
            classifier = EnsembleClassifier(classes=classes)
            classifier.train(training_data=training_fn, verbose=self.all_classifiers)
        logger.info('=' * 80)
        logger.info('best:')
        classifier.get_training_best()
        logger.info('=' * 80)
        logger.info('coverage: %.02f%%', classifier.get_training_coverage() * 100)
        if self.all_classifiers:
            logger.info('Aborting query with all classifiers.')
            sys.exit(0)

        # Find day with best score.
        logger.info('=' * 80)
        best_day_score, best_day_data = best_day
        logger.info('best_day_score: %s', best_day_score)
        logger.info('best_day_data:')
        pprint(best_day_data, indent=4)
        logger.info('best date: %s', best_date)
        logger.info('last full day: %s', last_full_day)
        last_full_day_date = last_full_day[0]
        if self.yes is None and abs((last_full_day_date - date.today()).days) > 1:
            if input(f'Last full day is {last_full_day_date}, which is over 1 day ago. Continue? [yn]:').lower()[0] != 'y':
                sys.exit(1)

        # Generate query sets for each variable metric. Base them on the best day, and incrementally change them from there, to avoid drastic changes
        # which may have harmful medical side-effects.
        logger.info('=' * 80)
        logger.info('ranges:')
        for _name, _range in sorted(column_ranges.items(), key=lambda o: o[0]):
            print('Column:', _name, _range)
        _, best_data = last_full_day
        queries = [] # [(name, description, data)]
        query_name_list = sorted(column_values)
        if self.only_attribute:
            query_name_list = [self.only_attribute]
        for name in query_name_list:

            if name == CLASS_ATTR_NAME:
                continue

            if name in column_predictables and not column_predictables[name]:
                continue

            logger.info('Query attribute name: %s', name)
            if isinstance(list(column_values[name])[0], Nom):
                logger.info('Nominal attribute.')
                # Calculate changes for a nominal column.
                for direction in column_nominals[name]:
                    query_value = direction = Nom(direction)
                    new_query = copy.deepcopy(best_data)
                    new_query[name] = direction
                    best_value = best_data.get(name, sorted(column_nominals[name])[0])
                    if best_value != query_value:
                        description = f'{name}: change from {best_value} -> {query_value}'
                        logger.info('\t%s', description)
                        queries.append((name, description, new_query))
            else:
                # Calculate changes for a numeric column.
                logger.info('Numeric attribute.')
                if not column_ranges.get(name):
                    logger.info('Has no column ranges. Skipping.')
                    continue
                _min, _max, _step = column_ranges[name]
                assert _min < _max, 'Invalid min/max!'
                if self.only_attribute or name in ('bed'):
                    # Check every possible value.
                    _value = _min
                    while _value <= _max:
                        logger.info('Checking query %s=%s.', name, _value)

                        new_query = copy.deepcopy(best_data)

                        # If our best day starting point is missing this metric, then use the mean.
                        _mean = None
                        if name not in new_query:
                            new_query[name] = sum(column_values[name], Num(0.0)) / len(column_values[name])
                            _mean = copy.deepcopy(new_query[name])

                        # Skip the hold case.
                        if _value == best_data.get(name, _mean):
                            logger.info('Hold case. Skipping.')
                            continue

                        new_query[name].value = _value
                        if best_data.get(name, _mean) != new_query[name]:
                            logger.info('\tallowable range min/max/step: %s %s %s', _min, _max, _step)
                            description = f'{name}: change from {best_data.get(name, _mean)} -> {new_query[name].value}'
                            logger.info('\t%s', description)
                            assert _min <= new_query[name].value <= _max
                            queries.append((name, description, new_query))

                        _value += _step
                else:
                    # Check only a relative change.
                    for direction in [-1, 1]:
                        new_query = copy.deepcopy(best_data)

                        # If our best day starting point is missing this metric, then use the mean.
                        _mean = None
                        if name not in new_query:
                            new_query[name] = sum(column_values[name], Num(0.0)) / len(column_values[name])
                            _mean = copy.deepcopy(new_query[name])

                        new_query[name].value += direction * _step
                        new_query[name].value = min(new_query[name].value, _max)
                        new_query[name].value = max(new_query[name].value, _min)
                        if best_data.get(name, _mean) != new_query[name]:
                            logger.info('\tallowable range min/max/step: %s %s %s', _min, _max, _step)
                            description = f'{name}: change from {best_data.get(name, _mean)} -> {new_query[name]}'
                            logger.info('\t%s', description)
                            queries.append((name, description, new_query))

            # Re-evaluate the current state.
            new_query = copy.deepcopy(best_data)
            description = f'{name}: hold at {best_data.get(name, _mean)}'
            queries.append((name, description, new_query))

        if save:
            logger.info('Saving classifier to %s...', self.classifier_fn)
            classifier.save(self.classifier_fn)
            logger.info('Classifier saved to %s.', self.classifier_fn)

        # Score each query.
        logger.info('=' * 80)
        total = len(queries)
        i = 0
        final_recommendations = [] # [(predicted change, old score, new score, description, name)]
        final_scores = {} # {name: (best predicted change, description)}
        for name, description, query_data in queries:
            i += 1
            logger.info('Running query %i of %i...', i, total)
            new_arff = arff.copy(schema_only=True)
            new_arff.relation = 'optimizer-query'
            query_data[CLASS_ATTR_NAME] = MISSING

            for _column, _controllable in column_learnables.items():
                if not _controllable and _column in query_data:
                    del query_data[_column]

            logger.info('query_data: %s', sorted(query_data))
            new_arff.append(query_data)
            logger.info('$' * 80)
            logger.info('predicting...')
            predictions = list(classifier.predict(new_arff, tolerance=TOLERANCE, verbose=1, cleanup=0))
            logger.info('\tdesc: %s', description)
            logger.info('\tpredictions: %s', predictions)
            if predictions:
                score_change = predictions[0].predicted
                logger.info('\tscore change: %.02f', score_change)
            else:
                logger.warning('No predictions found? Possible bug in data or predictor?')
                score_change = 0
            final_recommendations.append((score_change, 0, 0, description, name))
            final_scores.setdefault(name, (-1e999999999999, None))
            final_scores[name] = max(final_scores[name], (score_change, description))

        # Show top predictors.
        logger.info('=' * 80)
        logger.info('best predictors:')
        best_names = classifier.get_best_predictors(tolerance=TOLERANCE, verbose=True)
        logger.info(best_names)
        seed_date = last_full_day[0]

        # Show final top recommendations by attribute.
        print('=' * 80)
        print(f'recommendations by attribute based on date: {seed_date}')
        final_recommendations.sort(key=lambda o: (o[4], o[0]))
        i = 0
        digits = len(str(len(final_recommendations)))
        for change, _old_score, _new_score, description, name in final_recommendations:
            i += 1
            best_score_change, best_description = final_scores[name]
            if description != best_description:
                continue
            print(('\t%0' + str(digits) + 'i %s => %.06f') % (i, description, change))

        # Show final top recommendations by best change.
        final_recommendations.sort()

        print('=' * 80)
        print(f'Evening recommendations by change based on date: {seed_date}')
        print_recommendation(final_recommendations, final_scores, typ=EV)

        print('=' * 80)
        print(f'Morning recommendations by change based on date: {seed_date}')
        print_recommendation(final_recommendations, final_scores, typ=MN)

        print('=' * 80)
        print(f'Other recommendations by change based on date: {seed_date}')
        print_recommendation(final_recommendations, final_scores, typ=OTHERS)

        self.write_report(final_recommendations, final_scores)

        return final_recommendations, final_scores

    def write_report(self, recommendations, scores):
        report_dir = os.path.join(BASE_REPORTS_DIR, str(date.today()))
        os.makedirs(report_dir, exist_ok=True)
        fn = os.path.join(report_dir, 'ensemble.csv')
        logger.info('Writing analysis report to %s.', fn)
        with open(fn, 'w', encoding='ascii') as fout:
            fieldnames = [
                'name',
                'recommended action',
                'recommended value',
                'expected change',
            ]
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writerow(dict(zip(fieldnames, fieldnames)))
            for change, _old_score, _new_score, description, name in recommendations:
                best_score_change, best_description = scores[name]
                if description != best_description:
                    continue
                best_description = best_description.split(':')[-1].strip()
                if ' at ' in best_description:
                    best_action, best_value = best_description.split(' at ')
                elif ' from ' in best_description:
                    best_action, best_value = best_description.split(' from ')
                else:
                    raise Exception(f'Invalid description: {best_description}')
                writer.writerow({
                    'name': name,
                    'recommended action': best_action,
                    'recommended value': best_value,
                    'expected change': round(change, 2),
                })
        logger.info('Wrote analysis report to %s.', fn)


def print_recommendation(recs, scores, typ=None):
    i = len(recs) + 1
    digits = len(str(len(recs)))
    for change, _old_score, _new_score, description, name in recs:
        i -= 1
        if typ:
            if typ == EV and EV not in name:
                continue
            if typ == MN and MN not in name:
                continue
            if typ == OTHERS and (EV in name or MN in name):
                continue
        best_score_change, best_description = scores[name]
        if description != best_description:
            continue
        print(('\t%0' + str(digits) + 'i %s => %.06f') % (i, description, change))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyzes daily routine features to optimize your routine.')
    parser.add_argument('fn', help='Filename of ODS file containing data.')
    parser.add_argument('--only-attribute', default=None, help='If given, only predicts the effect of this one attribute. Otherwise looks at all attributes.')
    parser.add_argument('--stop-on-error', action='store_true', default=False, help='If given, halts at first error. Otherwise shows a warning and continues.')
    parser.add_argument('--no-train', action='store_true', default=False, help='If given, skips training and re-uses last trained classifier.')
    parser.add_argument(
        '--score-field-name', default=None, help=f'The name of the field containing the score to maximize. Default is "{DEFAULT_SCORE_FIELD_NAME}".'
    )
    parser.add_argument(
        '--all-classifiers',
        action='store_true',
        default=False,
        help='If given, trains all classifiers, even the crappy ones. Otherwise, only uses the known best.'
    )
    parser.add_argument(
        '--calculate-pcc', action='store_true', default=False, help='If given, calculates the Pearson correlation coefficient for all attributes.'
    )
    parser.add_argument(
        '--plot-attributes',
        default='',
        help='Comma-delimited list of columns names to plot, where x-axis is the column value and y-axis is the score value mean.'
    )
    parser.add_argument('--yes', default=None, action='store_true', help='Enables non-interactive mode and assumes yes for any interactive yes/no prompts.')
    args = parser.parse_args()
    o = Optimizer(**args.__dict__)
    o.analyze()
