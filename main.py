# main
import time

from processors import *
from utils import *
from statistics import *
import matplotlib.pyplot as plt
import logging
from parameters import *


def compute_and_save_all():
    for par in parameters:
        compute_and_save(par,'Results/'+par+"/")


def display_all():
    for par in parameters:
        display_results(par,'Results/'+par+"/",False)
        

def display_all_boxplot():
    for par in parameters:
        display_results(par,'Results/'+par+"/",True)


start_time = time.process_time_ns()

#generate_daggen()
#genTaskFiles()
compute_and_save_all()
display_all()
display_all_boxplot()


end_time = time.process_time_ns()

print(f"Finished computing in {(end_time-start_time)/(10**9):.3f}s")

