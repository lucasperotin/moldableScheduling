
# All the numerics we are going to use.

from math import *
from models import *

MODEL_LIST=[GeneralModel(),AmdahlModel(),CommunicationModel(),RooflineModel(),Power25Model(),Power50Model(),Power75Model(),Power100Model()]
#MODEL_LIST=[GeneralModel()]
Heuristics=["ICPP22", "TOPC24", "Fair","minTime"]
#Heuristics=["Amin", "CP"]
parameters=["Regular","Density","Fat","n","P", "Priority","Jump"]  
#parameters=["n"]

nb_iterations =50 #Number of experients for any set of parameters
alpha=0
mu=0


p_bounds=[1,640] 
USEWDAG=True #If set to true, the w used will be the one generated by daggen. If set to False, the w will be randomly generated as below
w_bounds=[2,5] #Between 10^2 and 10^5 
d_bounds=[-3,-0.5] #Between 10^-3 and 10^-0.5
c_bounds=[-6,-2] #Between 10^-6 and 10^-2



#This should be extremely flexible, you can add values, remove values, PMain don't have to be in p_list = [256, 384, 512, 640, 768, 896, 1024], etc
priority_list=["FIFO","procs","area","length"]
priorityMain="FIFO"
p_list = [256, 384, 512, 640, 768, 896, 1024]
PMain= 640
logP=False # For display, set to True if you test like [256,512,1024,2048,4096]
n_list = [1000, 2000, 3000, 4000, 5000] 
nMain=3000
jump_list = [ 2, 3, 4, 5, 6, 7, 8]  
jumpMain=5
fat_list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
fatMain=0.7
density_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5,0.6] 
densMain=0.3
regular_list = [0, 0.2, 0.4, 0.6, 0.8, 1] 
regMain=0.5

