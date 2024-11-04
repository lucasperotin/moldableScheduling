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





def generate_task(i):
    """Generate a task based on the boundaries written in numerics"""
    name="Task"+str(i)
    w = 10 ** uniform(w_bounds[0], w_bounds[1])
    p = randint(p_bounds[0], p_bounds[1])
    d = 10 ** uniform(d_bounds[0], d_bounds[1])
    c = 10 ** uniform(c_bounds[0], c_bounds[1])
    return Task(name,w, p, d, c)


def generate_n_tasks(n):
    """Generate a list of n tasks based on the boundaries written in numerics"""
    output = []
    for i in range(n):
        output += [generate_task(i)]
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
            nodes = generate_n_tasks(n)  
            file_path = os.path.join(subfolder, f"{i}.csv")
            
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                j=0
                writer.writerow(['w', 'p', 'd', 'c'])
                for task in nodes:
                    name=task.get_name()
                    w = task.get_w()
                    p = task.get_p()
                    d = task.get_d()
                    c = task.get_c()
                    writer.writerow([name, str(w), str(p), str(d), str(c)])
                    j+=1

    print("Task files generation completed.")

def load_nodes_from_csv(file):
    """Loads a set of nodes from a csv file"""
    f = open(file, 'r', newline='')
    reader = csv.reader(f)
    nodes = []
    i=0
    for row in reader:
        if row[0] != 'w':
            name=str(row[0])
            w = float(row[1])
            p_tild = float(row[2])
            d = float(row[3])
            c = float(row[4])
            nodes += [Task(name,w, p_tild, d, c)]
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
    elif variation_parameter == 'Priority':
        variation_list=priority_list
        
        
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
                      f" instance {(i+1):2d}/{nb_iterations},"
                      f" parameter {k + 1:2d}/{nbpar},"
                      f" ETA: {int(eta)}s,"
                      f" variation : {variation_parameter}")
                num += 1
                
                if variation_parameter == 'n':
                    daggen_file = "DAGGEN/" + variation_parameter + "_variation/" + variation_parameter + "=" + \
                                  str(n_list[k]) + "/" + str(i) + ".csv"
                    node_file = "TASKS/n=" + str(n_list[k]) + "/" + str(i) + ".csv"
                elif variation_parameter == 'P' or variation_parameter=="Priority":
                    daggen_file = "DAGGEN/n_variation/n="+(str)(nMain)+"/" + str(i) + ".csv"
                    node_file = "TASKS/n="+(str)(nMain)+"/" + str(i) + ".csv"
                else:
                    daggen_file = "DAGGEN/" + variation_parameter + "_variation/" + variation_parameter + "=" + \
                                  str(variation_list[k]) + "/" + str(i) + ".csv"
                    node_file = "TASKS/n="+(str)(nMain)+"/" + str(i) + ".csv"
                
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
                    
                if variation_parameter =="Priority":
                    priority=variation_list[k]
                else:
                    priority=priorityMain
                    

                task_graph = Graph(nodes, edges)
                processors = Processors(P_tild)

                logging.debug("\nmodel : " + model.name,variation_parameter + " = " + str(variation_list[k]) + ", file :" + str(i))

                speedup_model = model
                
                row=[str(variation_list[k])]
                amin=task_graph.get_A_min(P_tild, speedup_model)
                #print(format_scientific(amin / P_tild))
                cpmin=task_graph.get_C_min(P_tild,speedup_model)
                #print(format_scientific(cpmin))
                time_opt = max(amin,cpmin)
                
                priority=priorityMain
                
                for heu in Heuristics:
                    if (heu=="ICPP22"):
                        row+=[str(processors.online_scheduling_algorithm(task_graph, 1, alpha=alpha_tild,
                                                                             mu_tild=mu_tild
                                                                             , priority= priority, speedup_model=speedup_model, P_tild=P_tild
                                                                             , version=0))]
                    elif (heu=="TOPC24"):
                        row+=[str(processors.online_scheduling_algorithm(task_graph, 1, alpha=alpha_tild,
                                                                              mu_tild=mu_tild
                                                                             , priority= priority, speedup_model=speedup_model, P_tild=P_tild
                                                                             , version=1))]
                        
                    elif(heu=="minTime"):
                        row+=[str(processors.online_scheduling_algorithm(task_graph, 2, alpha=alpha_tild,
                                                                          mu_tild=mu_tild
                                                                         , priority= priority, speedup_model=speedup_model, P_tild=P_tild
                                                                         , version=0))]
                    else:
                        print("Error : unknown Heuristic")
                        os._exit()
                        
                # print("end"
                row+=[str(time_opt)]
                row+=[str(amin/P_tild)]
                row+=[str(cpmin)]
                writer.writerow(row)
        f.close()



def display_results(variation_parameter, result_directory,boxplot):
            
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
    elif variation_parameter == 'Priority':
        variation_list=priority_list
        
    nbpar=len(variation_list)
    nbheur=len(Heuristics)
    for model in model_list:
        HeurResults=[[] for k in range(nbheur)]
        BoundResults=[[] for k in range(2)]
        for k in range(nbheur):
            HeurResults[k] = [[] for i in range(nbpar)]
        for k in range(2):
            BoundResults[k] = [[] for i in range(nbpar)]
            
        f = open(result_directory + model.name + "/all.csv", newline='')
        reader = csv.reader(f)
        for row in reader:
            if row[0] != variation_parameter:
                for k in range (nbpar):
                    if (row[0]==(str) (variation_list[k])):
                        index=k
                for k in range(nbheur):
                    HeurResults[k][index]+=[float(row[k+1]) / float(row[nbheur+1])]
                BoundResults[0][index]+=float(row[nbheur+2])
                BoundResults[1][index]+=float(row[nbheur+3])
        f.close()
        if boxplot:
            # Add reversed figure for "Priority" variation parameter
            if variation_parameter == "Priority":
                fig, ax = plt.subplots(figsize=(12, 6))
                positions = np.arange(1, nbheur + 1)
                width = 0.8 / len(variation_list)
                
                for k, priority in enumerate(variation_list):
                    boxplot_data = [HeurResults[i][k] for i in range(nbheur)]
                    bp = ax.boxplot(boxplot_data, positions=positions + (k - (len(variation_list)-1)/2) * width, 
                                    widths=width, patch_artist=True, 
                                    whis=[10, 90],  # Set whiskers to 10th and 90th percentiles
                                    medianprops={'color': 'black', 'linewidth': 1.5},
                                    boxprops={'facecolor': plt.cm.Set3(k / len(variation_list)), 'edgecolor': 'black'},
                                    whiskerprops={'color': 'black', 'linewidth': 1.5},
                                    capprops={'color': 'black', 'linewidth': 1.5})
                    
                    # Add mean markers (stars)
                    means = [np.mean(data) for data in boxplot_data]
                    ax.scatter(positions + (k - (len(variation_list)-1)/2) * width, means, 
                               marker='*', color='red', s=100, zorder=3)
        
                ax.set_xlabel("Heuristics")
                ax.set_ylabel("Normalized Makespan")
                ax.set_title(f"{model.name} - Priority Comparison")
                ax.set_xticks(positions)
                ax.set_xticklabels(Heuristics)
                
                # Add legend for priorities
                legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=plt.cm.Set3(k / len(variation_list)), 
                                                 edgecolor='black', label=priority) 
                                   for k, priority in enumerate(variation_list)]
                ax.legend(handles=legend_elements, loc='upper right')
        
                plt.tight_layout()
                plt.savefig(result_directory + "Priority_" + model.name + "_boxplot")
                plt.close()
                
            else:
                
                fig, ax = plt.subplots(figsize=(12, 6))
                positions = np.arange(1, nbpar + 1)
                width = 0.8 / nbheur
                
                for k in range(nbheur):
                    boxplot_data = [HeurResults[k][i] for i in range(nbpar)]
                    bp = ax.boxplot(boxplot_data, positions=positions + (k - (nbheur-1)/2) * width, 
                                    widths=width, patch_artist=True, 
                                    whis=[10, 90],  # Set whiskers to 10th and 90th percentiles
                                    medianprops={'color': 'black', 'linewidth': 1.5},
                                    boxprops={'facecolor': plt.cm.Set3(k / nbheur), 'edgecolor': 'black'},
                                    whiskerprops={'color': 'black', 'linewidth': 1.5},
                                    capprops={'color': 'black', 'linewidth': 1.5})
                    
                    # Add mean markers (stars)
                    means = [np.mean(data) for data in boxplot_data]
                    ax.scatter(positions + (k - (nbheur-1)/2) * width, means, 
                               marker='*', color='red', s=100, zorder=3)
            
                ax.set_xlabel(variation_parameter)
                ax.set_ylabel("Normalized Makespan")
                ax.set_title(model.name)
                ax.set_xticks(positions)
                ax.set_xticklabels(variation_list)
                
                
                # Add legend (without mean star)
                legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=plt.cm.Set3(k / nbheur), 
                                                 edgecolor='black', label=Heuristics[k]) 
                                   for k in range(nbheur)]
                ax.legend(handles=legend_elements, loc='upper right')
            
                plt.tight_layout()
                plt.savefig(result_directory + variation_parameter + "_" + model.name + "_boxplot")
                plt.close()
            
        else:
            f = open(result_directory + model.name + "/mean.csv", 'w', newline='')
            writer = csv.writer(f)
            mean_Heurs = [[] for i in range (nbheur)]
            mean_Bound=[[] for i in range(2)]
            
            for k in range(nbpar):
                row=[variation_list[k]]
                for i in range(nbheur):
                    mean_Heurs[i]+=[mean(HeurResults[i][k])]
                    row+=[mean(HeurResults[i][k])]
                mean_Bound[0]+=[mean(BoundResults[0][k])]
                mean_Bound[1]+=[mean(BoundResults[1][k])]
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
            if variation_parameter == "P" and logP:
                plt.xscale('log')
            p_values = sorted(set(variation_list))
            plt.xticks(p_values)
            plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
            plt.gca().xaxis.set_tick_params(which='minor', size=0)
            plt.gca().xaxis.set_tick_params(which='minor', width=0)
    
    
            plt.savefig(result_directory + variation_parameter + "_" + model.name)
            plt.close()
            
            LabelBounds=["Area","CriticalPath"]
            for k in range(2):
                plt.plot(variation_list, mean_Bound[k], label=LabelBounds[k])
            plt.xlabel(variation_parameter)
            plt.legend()
            plt.ylabel("Value")
            plt.title(model.name)
            if variation_parameter == "P" and logP:
                plt.xscale('log')
            p_values = sorted(set(variation_list))
            plt.xticks(p_values)
            plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
            plt.gca().xaxis.set_tick_params(which='minor', size=0)
            plt.gca().xaxis.set_tick_params(which='minor', width=0)
    
    
            plt.savefig(result_directory + variation_parameter + "_" + model.name+ "bounds")
            plt.close()

