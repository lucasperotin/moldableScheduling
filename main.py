# main
import time

from processors import *
from utils import *
from statistics import *
import matplotlib.pyplot as plt
import logging


def compute_and_save_all(k):
#     compute_and_save('regular','Results_V1/Regular/',k,version=0)
#     compute_and_save('density','Results_V1/Density/',k,version=0)
#     compute_and_save('fat','Results_V1/Fat/',k,version=0)
#     compute_and_save('jump','Results_V1/Jump/',k,version=0)
#     compute_and_save('n', 'Results_V1/n/', k, version=0)
#     compute_and_save('p', 'Results_V1/P/', k, version=0)
    compute_and_save('regular','Results_V3/Regular/',k,version=1)
    compute_and_save('density','Results_V3/Density/',k,version=1)
    compute_and_save('fat','Results_V3/Fat/',k,version=1)
    compute_and_save('jump','Results_V3/Jump/',k,version=1)
    compute_and_save('n', 'Results_V3/n/', k, version=1)
    compute_and_save('p', 'Results_V3/P/', k, version=1)


def display_all():
    display_results('regular','Results_V3/Regular/')
    display_results('density','Results_V3/Density/')
    display_results('fat','Results_V3/Fat/')
    display_results('jump','Results_V3/Jump/')
    display_results('n', 'Results_V3/n/')
    display_results('p', 'Results_V3/P/')

    
#display_results('regular','Results_V1/Regular/')
#display_results('density','Results_V1/Density/')
#display_results('fat','Results_V1/Fat/')
#display_results('jump','Results_V1/Jump/')
#display_results('n', 'Results_V1/n/')
#display_results('p', 'Results_V1/P/')

# Displaying the results for the second version of the algorithm
    
nb_iterations = 1

start_time = time.process_time_ns()

#generate_daggen(31)
#genTaskFiles(30)
compute_and_save_all(nb_iterations)
display_all()

# Computations for the first version of the processor allocation algorithm (minimizing alpha)

# compute_and_save('regular','Results_V1/Regular/',nb_iterations,version=0)
# compute_and_save('density','Results_V1/Density/',nb_iterations,version=0)
# compute_and_save('fat','Results_V1/Fat/',nb_iterations,version=0)
# compute_and_save('jump','Results_V1/Jump/',nb_iterations,version=0)
# compute_and_save('n', 'Results_V1/n/', nb_iterations, version=0)
#compute_and_save('p', 'Results_V1/P/', nb_iterations, version=0)

# Computations for the second version of the processor allocation algorithm (minimizing beta)

end_time = time.process_time_ns()

print(f"Finished computing in {(end_time-start_time)/(10**9):.3f}s")

# Displaying the results for the first version of the algorithm

#display_results('regular','Results_V1/Regular/')
#display_results('density','Results_V1/Density/')
#display_results('fat','Results_V1/Fat/')
#display_results('jump','Results_V1/Jump/')
#display_results('n', 'Results_V1/n/')
#display_results('p', 'Results_V1/P/')

# Displaying the results for the second version of the algorithm

#display_results('regular','Results_V3/Regular/')
#display_results('density','Results_V3/Density/')
#display_results('fat','Results_V3/Fat/')
#display_results('jump','Results_V3/Jump/')
#display_results('n', 'Results_V3/n/')
#display_results('p', 'Results_V3/P/')

# To compare the two version of the processor allocation algorithm

# display_multiple_results("V1","V3","Density","Merging_V1_and_V3")
# display_multiple_results("V1","V3","Fat","Merging_V1_and_V3")
# display_multiple_results("V1","V3","Jump","Merging_V1_and_V3")
# display_multiple_results("V1","V3","Regular","Merging_V1_and_V3")
# display_multiple_results("V1", "V3", "n", "Merging_V1_and_V3")
# display_multiple_results("V1", "V3", "p", "Merging_V1_and_V3")

# To compare the two version of the processor allocation algorithm with standard parameters

# display_results_boxplot("V1", "V3", "Merging_V1_and_V3")

