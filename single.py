
import time

# A bunch of useful function to generate task graph and manipulate csv files.

from task import *
from random import *
import csv
import codecs
from numerics import *
from processors import *
from statistics import *
import matplotlib.pyplot as plt
from logging import log
from model import *
MODEL_LIST = [Power75Model()]




def generate_task(i,w_bounds, p_bounds, alpha_d_bounds, r_d_bounds, alpha_c_bounds, r_c_bounds):
    """Generate a task based on the boundaries written in numerics"""
    w = uniform(w_bounds[0], w_bounds[1])
    p = randint(p_bounds[0], p_bounds[1])
    d = uniform(alpha_d_bounds[0], alpha_d_bounds[1]) / \
        10 ** (randint(r_d_bounds[0], r_d_bounds[1]))
    c = uniform(alpha_c_bounds[0], alpha_c_bounds[1]) * \
        2 ** (randint(r_c_bounds[0], r_c_bounds[1]))
    return Task(i, w, p, d, c)


def generate_n_tasks(n, w_bounds, p_bounds, alpha_d_bounds, r_d_bounds, alpha_c_bounds, r_c_bounds):
    """Generate a list of n tasks based on the boundaries written in numerics"""
    output = []
    for i in range(n):
        output += [generate_task(i, w_bounds, p_bounds, alpha_d_bounds, r_d_bounds, alpha_c_bounds, r_c_bounds)]
    return output


def extract_dependencies_from_csv(file, utf_code="utf-16"):
    """
    This function extract dependencies from a DAGGEN Output under a csv format.

    For some files you may need to change "utf-16" by "utf-8" depending on the method you used to create the csv files
    from the DAGGEN algorithm.

    """
    edges = []
    f = codecs.open(file, "rb", utf_code)
    reader = csv.reader(f)
    for row in reader:
        if len(row) == 1 and row[0][0] != '/' and row[0][0] != 'd' and row[0][0] != '}':
            element = row[0][2:]
            i = 0
            while element[i] != ' ':
                i += 1
            first_node = int(element[0:i]) - 1
            while element[i] == "-" or element[i] == ">" or element[i] == " ":
                i += 1
            j = i
            while element[j] != ' ':
                j += 1
            second_node = int(element[i:j]) - 1
            edge = [first_node, second_node]
            edges += [edge]
    f.close()
    return edges


def generate_nodes_edges(n, w_bounds, p_bounds, alpha_d_bounds, r_d_bounds, alpha_c_bounds, r_c_bounds,
                         dependency_file):
    nodes = generate_n_tasks(n, w_bounds, p_bounds, alpha_d_bounds, r_d_bounds, alpha_c_bounds, r_c_bounds)
    edges = extract_dependencies_from_csv(dependency_file)
    for edge in edges:  # We need to pass from numbers to task objects
        edge[0] = nodes[edge[0]]
        edge[1] = nodes[edge[1]]
    return [nodes, edges]


def save_nodes_in_csv(n, w_bounds, p_bounds, alpha_d_bounds, r_d_bounds, alpha_c_bounds, r_c_bounds, file):
    """Saves a set of nodes and their parameters in a csv file"""
    nodes = generate_n_tasks(n, w_bounds, p_bounds, alpha_d_bounds, r_d_bounds, alpha_c_bounds, r_c_bounds)
    f = open(file, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['w', 'p', 'd', 'c'])
    for task in nodes:
        w = task.get_w()
        p = task.get_p()
        d = task.get_d()
        c = task.get_c()
        writer.writerow([str(w), str(p), str(d), str(c)])
    f.close()


def load_nodes_from_csv(file):
    """Loads a set of nodes from a csv file"""
    f = open(file, 'r', newline='')
    reader = csv.reader(f)
    nodes = []
    i=0
    for row in reader:
        if row[0] != 'w':
            i+=1
            w = float(row[0])
            p_tild = float(row[1])
            d = float(row[2])
            c = float(row[3])
            nodes += [Task((str)(i),w, p_tild, d, c)]
    return nodes



def compute_single():
    """

    :param variation_parameter: Can be : 'Fat', 'density', 'regular', 'jump', 'p', 'n'
    :param result_directory: A path to a directory containing 4 empty directories named 'Amdahl', 'communication',
                            'General', 'Roofline'.
    :param instances_nb: The number of different tasks graphs you want to run for each set of parameters. Must be picked
                         in the range [1,30]
    :return: Save the results in the corresponding directory depending on the speedup model
    """

    # Fixed parameters
    model_list = MODEL_LIST
    # name_list = ['Amdahl', 'Communication', 'General', 'Roofline']
    # mu_paper = [(1 - sqrt(8 * sqrt(2) - 11)) / 2, (23 - sqrt(313)) / 18, (33 - sqrt(738)) / 27, (3 - sqrt(5)) / 2]
    # alpha_paper = [(sqrt(2) + 1 + sqrt(2 * sqrt(2) - 1)) / 2, 4 / 3, 2, 1]
    P = 500
    p_tild=P
    version=1
    n = 20

    # for j in range(len(name_list)):
    num = 1
    model=RooflineModel()
    # Opening the result file
    f = open("result.csv", 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['P', 'Paper', 'Min Time', 'Time opt'])
    node_file = "testTasks.csv"
    daggen_file="testGraph.csv"
    nodes = load_nodes_from_csv(node_file)
    edges = extract_dependencies_from_csv(daggen_file)

    mu_tild = model.get_mu() 
    alpha_tild = model.get_alpha()
    task_graph = Graph(nodes, edges)
    processors = Processors(P)

               
    adjacency = task_graph.get_adjacency()
    #print(adjacency)

    speedup_model = model

    time_opt = task_graph.get_T_opt(p_tild, adjacency, speedup_model=speedup_model)
    print(time_opt)
                # print("start paper")
    time_algo_1 = processors.online_scheduling_algorithm(task_graph, 1, alpha=alpha_tild, adjacency=adjacency, mu_tild=mu_tild, speedup_model=speedup_model, P_tild=p_tild
                                                                     , version=version)
                # print("start min")
    time_algo_2 = processors.online_scheduling_algorithm(task_graph, 2, alpha=alpha_tild,
                                                                     adjacency=adjacency, mu_tild=mu_tild
                                                                     , speedup_model=speedup_model, P_tild=p_tild
                                                                     , version=version)
                # print("end")
                
    writer.writerow([str(P), str(time_algo_1), str(time_algo_2), str(time_opt)])
    f.close()
