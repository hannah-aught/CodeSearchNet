#!/usr/bin/env python
"""
Usage:
    confidence_intervals.py [options] MODEL_PATH VALID_DATA_PATH TEST_DATA_PATH
    confidence_intervals.py [options] MODEL_PATH

Standalone testing script

Options:
    -h --help                        Show this screen.
    --test-batch-size SIZE           The size of the batches in which to compute MRR. [default: 1000]
    --distance-metric METRIC         The distance metric to use [default: cosine]
    --run-name NAME                  Picks a name for the trained model.
    --quiet                          Less output (not one per line per minibatch). [default: False]
    --dryrun                         Do not log run into logging database. [default: False]
    --azure-info PATH                Azure authentication information file (JSON). Used to load data from Azure storage.
    --sequential                     Do not parallelise data-loading. Simplifies debugging. [default: False]
    --processes NUMBER               Number of processes to use during testing. [default: 16]
    --alpha VALUE                    Value to use for confidence level in the confidence interval. [default: 0.95]
    --debug                          Enable debug routines. [default: False]
"""

from pathlib import Path
from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath
import model_test as test
import scipy.stats as st
import numpy as np
from multiprocessing import Pool
import functools

def get_confidence_interval(data, alpha):
    return st.t.interval(alpha=alpha, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))

def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)
    valid_data_dir = test.expand_data_path(arguments['VALID_DATA_PATH'], azure_info_path)
    test_data_dir = test.expand_data_path(arguments['TEST_DATA_PATH'], azure_info_path)
    model_paths = RichPath.create(arguments['MODEL_PATH'], azure_info_path=azure_info_path).get_filtered_files_in_dir('*.pkl.gz')
    alpha = float(args['--alpha'])

    with Pool(int(arguments['--processes'])) as pool:
        results = pool.map(functools.partial(test.compute_evaluation_metrics, arguments=arguments, azure_info_path=azure_info_path, valid_data_dirs=valid_data_dir, test_data_dirs=test_data_dir, return_results=True, languages=['java'], test_valid=False), model_paths)

    docstring_mrrs = [x['java'][0] for x in results]
    func_name_mrrs = [x['java'][1] for x in results]

    docstring_confidence = get_confidence_interval(docstring_mrrs, alpha)
    func_name_confidence = get_confidence_interval(func_name_mrrs, alpha)

    print(f'{alpha*100}% confidence interval for mrr using docstring as the query: {docstring_confidence}')
    print(f'{alpha*100}% confidence interval for mrr using function name as the query: {func_name_confidence}')

if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])

