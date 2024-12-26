################################
# Verrecchia Thomas            #
# Summer - 2022                #
# Internship Kansas University #
################################

# This class implement our set of processors

from parameters import *
from graph import *
from task import Status
import csv
from math import *
from datetime import datetime
from models import *
from sortedcontainers import SortedList
import logging
from random import *

import os

def get_priority_key(task, priority, speedup_model=None):
    if priority == "length":
        return (-task.get_needed_time(), task.get_name())
    elif priority == "FIFO":
        return (task.get_discovery_time(), task.get_name())
    elif priority == "proc":
        return (-task.get_allocation(), task.get_name())
    elif priority == "area":
        return (-task.get_area(task.get_allocation(), speedup_model), task.get_name())
    else:
        return (0, task.get_name())  # Default case, sort by name only
    
    
class Processors:

    def __init__(self, nb_processors):
        self.nb_processors = nb_processors
        self.available_processors = nb_processors
        self.time = 0

    # Getters and Setters
    ############################################################

    def get_nb_processors(self):
        return self.nb_processors

    def get_available_processors(self):
        return self.available_processors

    def get_time(self):
        return self.time

    def set_nb_processors(self, value):
        self.nb_processors = value

    def set_available_processors(self, value):
        self.available_processors = value

    def set_time(self, value):
        self.time = value

    # Methods
    ############################################################

    def online_scheduling_algorithm(self, task_graph, alpha,
                                    P_tild, mu_tild, priority = "FIFO", speedup_model: Model = GeneralModel(), heuristic="minTime",save_in_logs=False):
        """"
        Given a task graph, this function calculate the time needed to complete every task of the task graph.
        It's the implementation of the algorithm 1 from the paper. Concerning the allocation_function :
        1 : allocate_processor_algo
        2 : allocate_processor_Min_time
        3 : allocate_processor_Min_area
        """
        logging.debug("  ---- Starting ----")
        logging.debug("Number of processors :", self.get_nb_processors())
        waiting_queue = set()  # Initialize a waiting queue Q
        process_list = []  # List of the task being processed
        nodes = task_graph.get_nodes()
        ratio=1

        # if save_in_logs:
        #     name = "logs/" + datetime.now().strftime("%m_%d_%Y-%H.%M.%S") + ".csv"
        #     log = open(name, 'w', newline='')
        #     writer = csv.writer(log)
        #     writer.writerow(['Time', 'Waiting Queue', 'Processors Queue', 'Number of available processors'])
        newratio=1
        for task in nodes:  # Insert all tasks without parents in the waiting queue
            if not task_graph.get_parents(nodes.index(task)):
                newratio=max(task.allocate_processor(heuristic,P_tild, mu_tild, alpha, ratio, speedup_model),newratio)
                task.set_needed_time(task.get_execution_time(task.get_allocation(), speedup_model))
                task.set_discovery_time(self.get_time())
                waiting_queue.add(task)
                task.set_status(Status.PROCESSING)
        
        if(newratio>ratio+0.000001):
            ratio=newratio
            for task in nodes:  # Insert all tasks without parents in the waiting queue
                if not task_graph.get_parents(nodes.index(task)):
                    newratio=max(task.allocate_processor(heuristic,P_tild, mu_tild, alpha, ratio, speedup_model),newratio)
                    task.set_needed_time(task.get_execution_time(task.get_allocation(), speedup_model))
                    task.set_discovery_time(self.get_time())
                    waiting_queue.add(task)
                    task.set_status(Status.PROCESSING)
                    
        if(newratio>ratio+0.000001):
            print("FATAL ERROR")
            os.exit()
            
        nbloo=0
        looptest=-1
        infloo=0
        while waiting_queue or process_list:
            #print(nbloo, self.get_time())
            #print("1")
            nbloo+=1
            # Cleaning of the processors
            if (self.get_time()==looptest):
                infloo+=1
                if (infloo>=100):
                    print("Error : infinite loop detected")
                    os._exit()
            else:
                looptest=self.get_time()
                infloo=0
            available_tasks = set()
            if process_list:
                task = min(process_list)
                process_list.remove(task)
                task.set_status(Status.PROCESSED)
                #print("Task "+(str) (task.get_name()) +" is completed "+(str) (self.get_time()))
                self.available_processors += task.get_allocation()
                for child in task_graph.get_children(nodes.index(task)):
                    nodes[child].set_nb_par_left(nodes[child].get_nb_par_left()-1)
                    if (nodes[child].get_nb_par_left()<0):
                        print (nodes[child].get_name(), len(task_graph.get_parents(child)), nodes[child].get_nb_par_left())
                        print("Error task already ready")
                        os._exit()
                        
                    elif (nodes[child].get_nb_par_left()==0):
                        
                        nodes[child].set_status(Status.AVAILABLE)
                        nodes[child].set_discovery_time(self.get_time())
                        available_tasks.add(nodes[child])

            # Processor allocation
            #print("2")
            newratio=1
            for task in available_tasks:
                newratio=max(task.allocate_processor(heuristic,P_tild, mu_tild, alpha, ratio, speedup_model),newratio)
                task.set_needed_time(task.get_execution_time(task.get_allocation(), speedup_model))
                waiting_queue.add(task)
                task.set_status(Status.PROCESSING)
                
            if(newratio>ratio+0.000001):
                ratio=newratio
                for task in waiting_queue:
                    newratio=max(task.allocate_processor(heuristic,P_tild, mu_tild, alpha, ratio, speedup_model),newratio)
                    task.set_needed_time(task.get_execution_time(task.get_allocation(), speedup_model))
                    waiting_queue.add(task)
                    task.set_status(Status.PROCESSING)
                    
            
            if(newratio>ratio+0.000001):
                print("FATAL ERROR")
                os.exit()
            # List Scheduling
            to_remove = set()
            #print("3")
            sorted_queue = sorted(waiting_queue, key=lambda task: get_priority_key(task, priority, speedup_model))
            #print(len(sorted_queue))
            for task in sorted_queue:                
                 # print((str) (task.get_name())+" "+(str) (self.get_available_processors())+" "+(str) (task.get_allocation()))
                if self.get_available_processors() >= task.get_allocation():
                    process_list.append(task)
                    to_remove.add(task)
                    task.set_starting_time(self.get_time())
                   # print("task " + (str) (task.get_name())+" w:"+(str) (task.get_w())+", p*: "+(str) (task.get_p())+" allocated "+(str) (task.get_allocation())+
                    #      " processors for execution time " + (str) (task.get_needed_time())+" Started at time "+(str)(self.get_time()))
                    self.available_processors -= task.get_allocation()
            for el in to_remove:
                waiting_queue.remove(el)
           # print("")
            # Incrementing time
            if process_list:
                next_task = min(process_list)
                self.time = next_task.get_needed_time() + next_task.get_starting_time()
        # if save_in_logs:
        #     log.close()

        #print(speedup_model.name, ratio, "AAAAAA")
        # Resetting the status and the clock of the processors
        task_graph.init_status()
        final_time = self.get_time()
        logging.debug("Total Execution time :", self.get_time(), "seconds")
        self.set_time(0)
        return final_time
