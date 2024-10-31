################################
# Verrecchia Thomas            #
# Summer - 2022                #
# Internship Kansas University #
################################
import time

# A bunch of useful function to generate task graph and manipulate csv files.

from task import *
from random import *
import csv
import codecs
from parameters import *
from processors import *
from statistics import *
import matplotlib.pyplot as plt
from logging import log
from model import *

import os
import shutil
import subprocess
import csv





def create_file_with_dirs(filepath, mode='w', newline=''):
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)
    return open(filepath, mode, newline=newline)


def generate_daggen():
    # Chemin absolu du répertoire de travail actuel
    current_dir = os.path.abspath(os.getcwd())
    
    # Chemin absolu du dossier DAGGEN
    daggen_output_dir = os.path.join(current_dir, "DAGGEN")
    
    # Chemin absolu de l'exécutable daggen
    daggen_executable = os.path.join(current_dir, "daggen-master","daggen")
    
    # Suppression et recréation du dossier DAGGEN
    if os.path.exists(daggen_output_dir):
        shutil.rmtree(daggen_output_dir)
    os.makedirs(daggen_output_dir)

    # Création des dossiers principaux
    main_folders = ["Density_variation", "Fat_variation", "n_variation", "Jump_variation", "Regular_variation"]
    for folder in main_folders:
        os.makedirs(os.path.join(daggen_output_dir, folder))

    # Création des sous-dossiers et exécution des commandes
    variations = {
        "Density_variation": density_list,
        "Fat_variation": fat_list,
        "n_variation": n_list,
        "Jump_variation": jump_list,
        "Regular_variation": regular_list
    }

    for main_folder, var_list in variations.items():
        print("Generating daggen : "+f"{main_folder.split('_')[0]}")
        for value in var_list:
            subfolder = f"{main_folder.split('_')[0]}={value}"
            subfolder_path = os.path.join(daggen_output_dir, main_folder, subfolder)
            os.makedirs(subfolder_path)
            
            for i in range(nb_iterations):
                output_file = os.path.join(subfolder_path, f"{i}.csv")
                if main_folder == "Density_variation":
                    cmd = [daggen_executable, "--dot", "--regular", str(regMain), "--density", str(value), "--jump", str(jumpMain), "-n", str(nMain), "--fat", str(fatMain), "-o", output_file]
                elif main_folder == "Fat_variation":
                    cmd = [daggen_executable, "--dot", "--regular", str(regMain), "--density", str(densMain), "--jump", str(jumpMain), "-n", str(nMain), "--fat", str(value), "-o", output_file]
                elif main_folder == "n_variation":
                    cmd = [daggen_executable, "--dot", "--regular", str(regMain), "--density", str(densMain), "--jump", str(jumpMain), "-n", str(value), "--fat", str(fatMain), "-o", output_file]
                elif main_folder == "Jump_variation":
                    cmd = [daggen_executable, "--dot", "--regular", str(regMain), "--density", str(densMain), "--jump", str(value), "-n", str(nMain), "--fat", str(fatMain), "-o", output_file]
                elif main_folder == "Regular_variation":
                    cmd = [daggen_executable, "--dot", "--regular", str(value), "--density", str(densMain), "--jump", str(jumpMain), "-n", str(nMain), "--fat", str(fatMain), "-o", output_file]
                
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)





def generate_task():
    """Generate a task based on the boundaries written in numerics"""
    w = 10 ** uniform(w_bounds[0], w_bounds[1])
    p = randint(p_bounds[0], p_bounds[1])
    d = 10 ** uniform(d_bounds[0], d_bounds[1])
    c = 10 ** uniform(c_bounds[0], c_bounds[1])
    return Task(w, p, d, c)


def generate_n_tasks(n):
    """Generate a list of n tasks based on the boundaries written in numerics"""
    output = []
    for i in range(n):
        output += [generate_task()]
    return output


def extract_dependencies_from_csv(file, utf_code="utf-8"):
    """
    This function extract dependencies from a DAGGEN Output under a csv format.

    For some files you may need to change "utf-16" by "utf-8" depending on the method you used to create the csv files
    from the DAGGEN algorithm.

    """
    edges = []
    f = codecs.open(file, "rb", utf_code)
    reader = csv.reader(f)
    values=[]
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
        elif len(row)==2:
            element = row[0][1:]
            i=0
            while element[i]!='"':
                i+=1
            i+=1
            j=i
            while element[j] != '"':
                j+=1
            node=(int) (element[i:j])
            values+=[node]
    f.close()
    
    return edges,values



def genTaskFiles():
    """
    Deletes everything in the TASKS folder, recreates subfolders for each n value,
    and saves a set of nodes and their parameters in csv files.
    """
    tasks_dir = "TASKS"
    
    # Delete everything in the TASKS folder
    if os.path.exists(tasks_dir):
        shutil.rmtree(tasks_dir)
    
    # Recreate the TASKS folder
    os.makedirs(tasks_dir)
    
    # Create subfolders for each n value
    for n in n_list:
        subfolder = os.path.join(tasks_dir, f"n={n}")
        os.makedirs(subfolder)
    
    # Generate task files
    for n in n_list:
        print(f"Generating speedups : n={n}")
        subfolder = os.path.join(tasks_dir, f"n={n}")
        for i in range(nb_iterations):
            nodes = generate_n_tasks(n)  # Assuming this function exists
            file_path = os.path.join(subfolder, f"{i}.csv")
            
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['w', 'p', 'd', 'c'])
                for task in nodes:
                    w = task.get_w()
                    p = task.get_p()
                    d = task.get_d()
                    c = task.get_c()
                    writer.writerow([str(w), str(p), str(d), str(c)])

    print("Task files generation completed.")

def load_nodes_from_csv(file):
    """Loads a set of nodes from a csv file"""
    f = open(file, 'r', newline='')
    reader = csv.reader(f)
    nodes = []
    i=0
    for row in reader:
        if row[0] != 'w':
            w = float(row[0])
            p_tild = float(row[1])
            d = float(row[2])
            c = float(row[3])
            nodes += [Task(w, p_tild, d, c)]
            i+=1
    return nodes


def compute_and_save(variation_parameter, result_directory):
    """

    :param variation_parameter: Can be : 'fat', 'density', 'regular', 'jump', 'p', 'n'
    :param result_directory: A path to a directory containing 4 empty directories named 'Amdahl', 'communication',
                            'General', 'Roofline'.
    """

    # Fixed parameters
    model_list = MODEL_LIST
    # name_list = ['Amdahl', 'Communication', 'General', 'Roofline']
    # mu_paper = [(1 - sqrt(8 * sqrt(2) - 11)) / 2, (23 - sqrt(313)) / 18, (33 - sqrt(738)) / 27, (3 - sqrt(5)) / 2]
    # alpha_paper = [(sqrt(2) + 1 + sqrt(2 * sqrt(2) - 1)) / 2, 4 / 3, 2, 1]

    if variation_parameter == 'P':
        variation_list=p_list
    elif variation_parameter=="n":
        variation_list=n_list   
    elif variation_parameter == 'Fat':
        variation_list=fat_list
    elif variation_parameter == 'Density':
        variation_list=density_list
    elif variation_parameter == 'Regular':
        variation_list=regular_list
    elif variation_parameter == 'Jump':
        variation_list=jump_list
        
        
    nbpar=len(variation_list)  
    # for j in range(len(name_list)):
    start_time = time.process_time_ns()
    num = 1
    for idx, model in enumerate(model_list):
        # Opening the result file
        firstline=[variation_parameter]
        for alg in Heuristics:
            firstline+=[alg]
        firstline+=["Topt"]
        f = create_file_with_dirs(result_directory + str(model.name) + "/all.csv", 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(firstline)

        for i in range(nb_iterations):
            for k in range(nbpar):
                pc = num / (nb_iterations * nbpar * len(MODEL_LIST))
                eta = ((time.process_time_ns() - start_time) / 1e9) * ((1 - pc) / pc)
                print(f"[{pc * 100:.2f} %]"
                      f" {model.name} model ({idx + 1}/{len(MODEL_LIST)}),"
                      f" instance {i:2d}/{nb_iterations},"
                      f" parameter {k + 1:2d}/{nbpar}"
                      f" ETA: {int(eta)}s")
                num += 1
                
                if variation_parameter == 'n':
                    daggen_file = "DAGGEN/" + variation_parameter + "_variation/" + variation_parameter + "=" + \
                                  str(n_list[k]) + "/" + str(i) + ".csv"
                    node_file = "TASKS/n=" + str(n_list[k]) + "/" + str(i) + ".csv"
                elif variation_parameter == 'P':
                    daggen_file = "DAGGEN/n_variation/n=500/" + str(i) + ".csv"
                    node_file = "TASKS/n=500/" + str(i) + ".csv"
                else:
                    daggen_file = "DAGGEN/" + variation_parameter + "_variation/" + variation_parameter + "=" + \
                                  str(variation_list[k]) + "/" + str(i) + ".csv"
                    node_file = "TASKS/n=500/" + str(i) + ".csv"
                
                nodes = load_nodes_from_csv(node_file)
                edges,ww = extract_dependencies_from_csv(daggen_file)
                if (USEWDAG):
                    for j in range(len(nodes)):
                        #print(nodes[i].get_w())
                        nodes[j].set_w(ww[j])
                        #print(nodes[i].get_w())
                #print(daggen_file)
                if (mu==0):
                    mu_tild = model.get_mu()
                else:
                    mu_tild=mu
                    
                if(alpha==0):
                    alpha_tild = model.get_alpha()
                else:
                    alpha_tild = alpha

                if variation_parameter == 'P':
                    P_tild = variation_list[k]
                else:
                    P_tild = PMain

                task_graph = Graph(nodes, edges)
                processors = Processors(P_tild)

                logging.debug("\nmodel : " + model.name,variation_parameter + " = " + str(variation_list[k]) + ", file :" + str(i))
                adjacency = task_graph.get_adjacency()

                speedup_model = model
                
                row=[str(variation_list[k])]
                time_opt = task_graph.get_T_opt(P_tild, adjacency, speedup_model=speedup_model)
                
                for heu in Heuristics:
                    if (heu=="ICPP22"):
                        row+=[str(processors.online_scheduling_algorithm(task_graph, 1, alpha=alpha_tild,
                                                                             adjacency=adjacency, mu_tild=mu_tild
                                                                             , speedup_model=speedup_model, P_tild=P_tild
                                                                             , version=0))]
                    elif (heu=="TOPC24"):
                        row+=[str(processors.online_scheduling_algorithm(task_graph, 1, alpha=alpha_tild,
                                                                             adjacency=adjacency, mu_tild=mu_tild
                                                                             , speedup_model=speedup_model, P_tild=P_tild
                                                                             , version=1))]
                        
                    elif(heu=="minTime"):
                        row+=[str(processors.online_scheduling_algorithm(task_graph, 2, alpha=alpha_tild,
                                                                         adjacency=adjacency, mu_tild=mu_tild
                                                                         , speedup_model=speedup_model, P_tild=P_tild
                                                                         , version=0))]
                        
                    else:
                        print("Error : unknown Heuristic")
                        os._exit()
                        
                # print("end")
                row+=[str(time_opt)]
                writer.writerow(row)
        f.close()



def display_results(variation_parameter, result_directory):
            
    model_list = MODEL_LIST
    # name_list = ["Amdahl", "Communication", "General", "Roofline"]
    
    if variation_parameter == 'P':
        variation_list=p_list
    elif variation_parameter=="n":
        variation_list=n_list   
    elif variation_parameter == 'Fat':
        variation_list=fat_list
    elif variation_parameter == 'Density':
        variation_list=density_list
    elif variation_parameter == 'Regular':
        variation_list=regular_list
    elif variation_parameter == 'Jump':
        variation_list=jump_list
        
    nbpar=len(variation_list)
    nbheur=len(Heuristics)
    for model in model_list:
        HeurResults=[[] for k in range(nbheur)]
        for k in range(nbheur):
            HeurResults[k] = [[] for i in range(nbpar)]
        f = open(result_directory + model.name + "/all.csv", newline='')
        reader = csv.reader(f)
        for row in reader:
            if row[0] != variation_parameter:
                for k in range (nbpar):
                    if (row[0]==(str) (variation_list[k])):
                        index=k
                for k in range(nbheur):
                    HeurResults[k][index]+=[float(row[k+1]) / float(row[nbheur+1])]
        f.close()
        f = open(result_directory + model.name + "/mean.csv", 'w', newline='')
        writer = csv.writer(f)
        mean_Heurs = [[] for i in range (nbheur)]
        
        for k in range(nbpar):
            row=[variation_list[k]]
            for i in range(nbheur):
                mean_Heurs[i]+=[mean(HeurResults[i][k])]
                row+=[mean(HeurResults[i][k])]
            writer.writerow(row)
            #print(mean_Paper)
        f.close()

        # Graphic parameters for the display
        
        for k in range(nbheur):
            plt.plot(variation_list, mean_Heurs[k], label=Heuristics[k])
        plt.xlabel(variation_parameter)
        plt.legend()
        plt.ylabel("Normalized Makespan")
        plt.title(model.name)
        if variation_parameter == "P":
            plt.xscale('log')
            p_values = sorted(set(variation_list))
            plt.xticks(p_values)
            plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
            plt.gca().xaxis.set_tick_params(which='minor', size=0)
            plt.gca().xaxis.set_tick_params(which='minor', width=0)


        plt.savefig(result_directory + variation_parameter + "_" + model.name)
        #plt.show()
        plt.close()



def display_results_boxplot(version1, version2, saving_directory):
    name_list = ["Amdahl", "Communication", "General", "Roofline"]
    parameters = ["Density", "fat", "Jump", "n", "p"]
    for name in name_list:
        Paper_V1 = []
        Paper_V2 = []
        Min_Time = []
        f = open("Results_" + version1 + "/P/" + name + "/all.csv", 'r', newline='')
        reader = csv.reader(f)
        for line in reader:
            if line[0] == "1500":
                Paper_V1 += [float(line[1]) / float(line[3])]
        f.close()
        f = open("Results_" + version2 + "/P/" + name + "/all.csv", 'r', newline='')
        reader = csv.reader(f)
        for line in reader:
            if line[0] == "1500":
                Paper_V2 += [float(line[1]) / float(line[3])]
                Min_Time += [float(line[2]) / float(line[3])]
        f.close()
        plt.boxplot([Paper_V1, Paper_V2, Min_Time])
        plt.xticks([1, 2, 3], ['Paper_' + version1, 'Paper_' + version2, 'Min Time'])
        plt.ylabel('Normalized Makespan')
        plt.savefig(saving_directory + "/" + name + "_Default_parameters.png")
        plt.show()
