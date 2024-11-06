################################
# Verrecchia Thomas            #
# Summer - 2022                #
# Internship Kansas University #
################################

# This class implement the well known concept of graph, in our case the nodes are tasks (see clas Task) and the graph is
# a directed acyclic graph (DAG)

from task import Task
import numpy as np
import logging
from task import Status

def format_scientific(number):
    # Format to scientific notation with 3 significant figures
    formatted = f"{number:.2e}"
    
    # Split into coefficient and exponent
    coeff, exp = formatted.split('e')
    
    # Remove leading '+' from exponent if present
    exp = exp.lstrip('+')
    
    # Format as x.yy*10^k
    return f"{coeff}*10^{exp}"

# In your print statement
class Graph:
    def __init__(self, nodes=None, edges=None):
        self._nodes = nodes or []  # a list of tasks [task1, task2, ...]
        self._edges = edges or []  # a list of edges [[task1, task2], [task3, task4], ...] where task1 is parent to task2
        self._children = []  # list of lists where _children[i] contains all children of task i
        self._parents = []   # list of lists where _parents[i] contains all parents of task i
        self._initialize_relationships()
        self.set_nb_par_left_for_all_tasks()

    def _initialize_relationships(self):
        if not self._nodes:
            return

        n = len(self._nodes)
        self._children = [[] for _ in range(n)]
        self._parents = [[] for _ in range(n)]

        for parent, child in self._edges:
            self._children[parent].append(child)
            self._parents[child].append(parent)
        
    def set_nb_par_left_for_all_tasks(self):
        for i, task in enumerate(self._nodes):
            num_parents = len(self._parents[i])
            task.set_nb_par_left(num_parents)
        
    def get_children(self, task):
        """Return a list of the children of a certain task. The argument 'task' takes an int value corresponding
        to the index of the task in the nodes list"""
        return self._children[task]

    def get_parents(self, task):
        """Return a list of the parents of a certain task. The argument 'task' takes an int value corresponding
        to the index of the task in the nodes list"""
        return self._parents[task]

    # Getters and setters (if still needed)
    def get_nodes(self):
        return self._nodes


    def set_nodes(self, value):
        self._nodes = value
        self._initialize_relationships()

    def set_edges(self, value):
        self._edges = value
        self._initialize_relationships()

    def get_A_min(self, P, speedup_model):
        """A_min is the sum of all the minimum area of each tasks"""
        A_min = 0
        for task in self.get_nodes():
            A_min += task.get_minimum_area(P, speedup_model)[0]
        return A_min
    def get_A_max(self,P,speedup_model):
        A_max=0
        for task in self.get_nodes():
            A_max += task.get_minimum_execution_time(P, speedup_model)[0]*task.get_minimum_execution_time(P, speedup_model)[1]
        return A_max

    def get_C_min(self, P, speedup_model):
        """C_min is the minimal execution time for the graph"""
        nodes = self.get_nodes()
        maximum_weight = 0
        weights = [0 for _ in range(len(nodes))]

        logging.debug("Selecting starting nodes...")

        free_nodes = []
        free_nodes_set = set()
        # Selecting the tasks without parents as starting points
        for index_task in range(len(nodes)):
            if not self.get_parents(index_task):
                free_nodes += [index_task]
                free_nodes_set.add(index_task)
        logging.debug("Calculating Optimal time...")

        idx = 0
        while idx < len(free_nodes):
            index_task = free_nodes[idx]
            idx += 1
            weight = nodes[index_task].get_minimum_execution_time(P, speedup_model)[0]
            p_weight = 0
            for index_parent in self.get_parents(index_task):
                if p_weight < weights[index_parent]:
                    p_weight = weights[index_parent]
            weight += p_weight
            if weight > maximum_weight:
                maximum_weight = weight
            weights[index_task] = weight

            for children in self.get_children(index_task):
                if children not in free_nodes_set:
                    free_nodes_set.add(children)
                    free_nodes.append(children)
        return maximum_weight

    def get_T_opt(self, P, speedup_model):
        """Return the inferior bound for T optimal for a given graph"""
        print(
            format_scientific(self.get_A_min(P, speedup_model) / P),
            format_scientific(self.get_A_max(P, speedup_model) / P),
            format_scientific(self.get_C_min(P, speedup_model))
        )
        output = max(self.get_A_min(P, speedup_model) / P, self.get_C_min(P, speedup_model))
        logging.debug("Optimal execution time :", output)
        return output

    def init_status(self):
        """Reset the status of each task in the graph"""
        for task in self.get_nodes():
            task.set_status(Status.BLOCKED)
