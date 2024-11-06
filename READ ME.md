
There are 2 importants files to modify

- parameters.py : Contains all the parameters of the simulation, 
	Let's first take a look at the first 3 lists:
	* MODEL_LIST : All different models. (I removed Power0 since it's equivalent to Amdahl). The descriptions can be found in the paper and internship reports. 
			8 models are currently available, [GeneralModel(),AmdahlModel(),CommunicationModel(),RooflineModel(),Power25Model(),Power50Model(),Power75Model(),Power100Model()].
			You can add your own model in models.py, you just need to define alpha, mu the time function and p_max that minimizes the execution time.
	* Heuristics : 5 Heuristics are available. This only impacts the allocation procedure. 
		- ICPP22 : See https://inria.hal.science/hal-03778405/document
		- TOPC24 : See https://arxiv.org/pdf/2304.14127	
		- Fair : New simple heuristic : Step 1, find p minimizing max(alpha,beta), step 2 : if p is to big, reduce to (3-sqrt(5))/2 P
		- minTime : minimize time
		- minArea : minimize area			(not recommended for experiments, heuristic is to bad)
	You can design your own heuristic, simply go to task.py, and find the allocate_processor function.
	* parameters : 7 parameters to experiment on. If you add an element on this list, it will generate instances to experiment on the effect of the variation of the parameter.
		- "Regular","Density","Fat", "Jump" : parameters for the shape of the graph, see https://github.com/frs69wq/daggen
		- "n" : number of tasks in the graph
		- "P" : number of processors available
		- "Priority" : priority rule. In the algorithm, at each step the tasks that are ready will be in a list, and the scheduler will go through the list to try to schedule them.
					      The priority argument decides how the list is sorted and what task will be chosen first by the scheduler. See later for more info on the different options

	These 3 lists will define our list of experiments, there will be graphs generated for any possible couples (model, parameters) and in thoses graphs all the heuristics will be represented,
	and the x axis will correspond to the different possible value of the parameter in parameters that is varied




	Next are a few variables to define for the general experiment
	* nb_iterations : gives the number of iteration for each set of parameters (Heuristic, model, parameter, parameter_value), where parameter_value is the value of the parameter in parameters you want to vary. Any point on the resulting graphs will be the average of all the iterations. I recommend using something in [30,100], less may not be enough to cover all possible scenario, more is a waste of time
	* alpha, mu. I heard you want to test different value of alpha and mu, if so you can set them here. Modifying this will change the behaviour of ICPP22 and TOPC24 Heuristics
	* USEWDAG : When generating graph in DAGGEN, it gives a length for each task. I left the option to generate the length of task synthetically instead, if turned to False
	* p_bounds, w_bounds, d_bounds, c_bounds : parameters of speedup functions. All models will use data generated here.

	Finally, given the parameters you cant to experiment on in parameters, here it how it wil vary:
	- All parameters xxx will be set up to the value specified in xxxMain, except the one that is varied in the experiments. 
	For example, if we are currently making P vary, for model= Amdahl, the code will launch the simulator for all heuristics, for Amdahl, with P=256, n=nMain, priority=priorityMain, ... and then will do the same with P=384, etc for all values in p_list.
	You may play with it, but I recommend making sure to keep a balance between the minimal critical path CP and the minimal Area Amin : if CP<<Amin, then minArea is optimal.
	If CP>>Amin, the minTime is optimal, neither of these scenario are interesting.
	To help you check this balance, I plotted CP and Amin for all experience


- Once you have set the parameters, go to main.py. It consists of 6 function :	
    	* generate_daggen()    # Generate the graphs (dependencies and length of each task) using DAGGEN. 
    	* genTaskFiles()       # Generate the speedup functions of task
    	* run_in_parallel(compute_and_save_wrapper, parameters)   # Do the simulations (suggestion, delete previous Results folder if you redo simulation : this function will either replace preexisting files if exists, or create new files)
    	* run_in_parallel(display_wrapper, parameters)            # Generate graphs in forms of lines, one line per heuristic. Will also generate figure for area/critical path
    	* run_in_parallel(display_boxplot_wrapper, parameters)    # Generate graphs but with boxplots
    	* generate_latex_report("results_visualization.tex")      # Organize all generated graphs in a latex file


	The code uses parallelism, by default on all different parameters you will experiment in. It is worth mentioning it also take cares about organizing the graph generated in a latex file, that consists of 5 Sections :
		- 1 : all plots with lines. For each couple (model, parameter) a graph is generated, x-axis is all different parameters, y-axis is normalized makespan (divided by lower bound), each line corresponds to a heuristic, average for all iteration
		- 2 : Same plot but with box plot, to show 10th, 25th, 75th and 90th percentile, with all extreme values as little circles
		- 3 : Critical Path and Lower bounds. If one is 10 times bigger than the other, probably you will want to change the parameters
		- 4 : Table of values, for any couple (heuristic, model) average of all experiments corresponding to this couple. Last line is an overall average on all models
		- 5 : Same with maximum, to check the theoretical results are true.



	
