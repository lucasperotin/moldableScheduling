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
    generate_daggen()    # Generate the graphs (dependencies and length of each task) using DAGGEN. 
    genTaskFiles()       # Generate the speedup functions of task
    run_in_parallel(compute_and_save_wrapper, parameters)   # Do the simulations (suggestion, delete previous Results folder if you redo simulation : this function will either replace preexisting files if exists, or create new files)
    run_in_parallel(display_wrapper, parameters)            # Generate graphs in forms of lines, one line per heuristic. Will also generate figure for area/critical path
    run_in_parallel(display_boxplot_wrapper, parameters)    # Generate graphs but with boxplots
    generate_latex_report("results_visualization.tex")      # Organize all generated graphs in a latex file