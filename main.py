import time
from multiprocessing import Pool
import os

from generate_latex import *
from processors import *
from utils import *
from statistics import *
import matplotlib.pyplot as plt
import logging
from parameters import *

def compute_and_save_wrapper(par):
    compute_and_save(par, f'Results/{par}/')

def display_wrapper(par):
    display_results(par, f'Results/{par}/', False)

def display_boxplot_wrapper(par):
    display_results(par, f'Results/{par}/', True)

def run_in_parallel(func, iterable):
    num_cores = min(os.cpu_count(), len(iterable))  # Use the minimum of available cores and number of parameters
    with Pool(num_cores) as p:
        p.map(func, iterable)

if __name__ == '__main__':
    #generate_daggen()
    #genTaskFiles()
    #run_in_parallel(compute_and_save_wrapper, parameters)
    #run_in_parallel(display_wrapper, parameters)
    #run_in_parallel(display_boxplot_wrapper, parameters)
    generate_latex_report(parameters, MODEL_LIST)