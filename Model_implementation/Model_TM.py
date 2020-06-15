#!/usr/bin/env python
# coding: utf-8

# In[30]:


#Import packages
import sys
try:
    import docplex.mp
except:
    raise Exception('Please install docplex. See https://pypi.org/project/docplex/')
    
from Routing_data import *
import pandas as pd
import ast 
import numpy as np


# In[31]:


#Define parameters of experiment

#Rescheduling period [tw1, tw2] for this problem, tw1 = 1:00 am and tw2 = 3:00am. 
#We convert all times to minutes
tw1 = 60.0
tw2 = 180.0

#Defines the data sheet (sheet number in train timetable)
scenario_counter = 5

#reads in train timetable data
data = pd.read_excel('TPP_data.xls', scenario_counter) 

#Big M
M = 1000000

#Safety Headway time (in min)
Hsp = 2


# In[32]:


edges_list = []
for route in R_in + R_out:
    for edge in route:
        edges_list.append(edge)
        
R_in_pairs = [[i,j] for i in R_in for j in R_in]
R_out_pairs = [[i,j] for i in R_out for j in R_out]


# In[35]:


#set of nodes, V
V = list(range(106))
V = V[1:]

#set of trains arriving to station within tw1 and tw2
Ih = [data['车次'][i] for i in range(0, len(data['车次'])) if tw1 <= data['Adjusted A (min)'][i] <= tw2]
Ih_pairs = [[i,j] for i in Ih for j in Ih if i != j]

#set of trains whose departure time from the station is later than tw2
Ir = [data['车次'][i] for i in range(0, len(data['车次'])) if tw1 <= data['Adjusted A (min)'][i] <= tw2 and data['D(min)'][i] > tw2]

#set of trains occupying platforms at tw1
Ip = [data['车次'][i] for i in range(0, len(data['车次'])) if data['Adjusted A (min)'][i] < tw1 < data['D(min)'][i] and data['A(min)'][i] != 0]

#set of all trains that will visit the station [tw1,tw2] 
I = Ip + Ih

#set of approach directions
H = [data['Entering from'][i] for i in range (0, len(data['Entering from'])) if data['D(min)'][i] >= tw1 and data['Adjusted A (min)'][i] <= tw2 and data['A(min)'][i] != 0]

#set of departure directions
G = [data['Exiting from'][i] for i in range (0, len(data['Entering from'])) if data['D(min)'][i] >= tw1 and data['Adjusted A (min)'][i] <= tw2 and data['A(min)'][i] != 0]

#Dict of trains and their approach directions
train_home = {train: home_signal for train, home_signal in zip(I, H)}

#Dict of trains and their departure directions
train_exit = {train: exit for train, exit in zip(I, G)}

#Arrival times of trains
Arvtime = [round(data['Adjusted A (min)'][i]) for i in range(0, len(data['车次'])) if tw1 <= data['Adjusted A (min)'][i] <= tw2]
At = {train:arrivaltime for train, arrivaltime in zip(Ih, Arvtime)}
for train in Ip:
    At.update({train: tw1})

#Scheduled departure time of trains
Scheduled_departure = [data['D(min)'][i] for i in range(0, len(data['车次'])) if tw1 <= data['Adjusted A (min)'][i] <= tw2]
Dt = {train:departuretime for train, departuretime in zip(Ih, Scheduled_departure)}

#set of platforms in the station
A = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'Dummy']

#Original platform allocation in previous assignment
O_platform = [data['股道/Platform track'][i] for i in range (0, len(data['股道/Platform track'])) if data['D(min)'][i] >= tw1 and data['Adjusted A (min)'][i] <= tw2 and data['A(min)'][i] != 0]
Op = {train: platform for train, platform in zip(I, O_platform)}

#set of platforms occupied by trains in set Ip
B = [data['股道/Platform track'][i] for i in range(0, len(data['车次'])) if data['Adjusted A (min)'][i] < tw1 < data['D(min)'][i] and data['A(min)'][i] != 0]

#release time associated with platform k
rx = [round(data['D(min)'][i]) for i in range(0, len(data['车次'])) if data['Adjusted A (min)'][i] < tw1 < data['D(min)'][i] and data['A(min)'][i] != 0]
release_time = []
for time in rx:
    release_time.append(time - tw1)    
rt = {platform:r for platform, r in zip (B, release_time)}


#Dwell time of trains at platforms
dt = [round(data['Dwell time (mins)'][i]) for i in range(0, len(data['车次'])) if tw1 <= data['Adjusted A (min)'][i] <= tw2]
dwell_time = {train: dwelltime for train, dwelltime in zip(Ih, dt)}
train_platform_release = {train:release for train, release in zip(Ip, release_time)}
dwell_time.update(train_platform_release)

#Allowable waiting time for trains at home signal (in this experiment, a uniform maximum waiting time of 3 mins is
#assigned to each train)
waiting_time = [3.0 for i in range(len(Ih))]
wt = {train:waitingtime for train, waitingtime in zip (Ih, waiting_time)}
for train in Ip:
    wt.update({train: 0})


# In[36]:


#Reading PM tasks data

PMTdata = pd.read_excel('Maintenance tasks.xlsx', 'PMT')

PM_tasks_list = [PMTdata['Maintenance task'][i] for i in range (0, len(PMTdata['Maintenance task'])) if tw1 <= PMTdata['Start time (min)'][i] <= tw2 or tw1 <= PMTdata['End time (min)'][i] <= tw2]
Ipm_list = [PMTdata['Start time (min)'][i] for i in range (0, len(PMTdata['Maintenance task'])) if tw1 <= PMTdata['Start time (min)'][i] <= tw2 or tw1 <= PMTdata['End time (min)'][i] <= tw2]
Fpm_list = [PMTdata['End time (min)'][i] for i in range (0, len(PMTdata['Maintenance task'])) if tw1 <= PMTdata['Start time (min)'][i] <= tw2 or tw1 <= PMTdata['End time (min)'][i] <= tw2]
Throat_edges = [PMTdata['Maintenance edges'][i] for i in range (0, len(PMTdata['Maintenance task'])) if tw1 <= PMTdata['Start time (min)'][i] <= tw2 or tw1 <= PMTdata['End time (min)'][i] <= tw2]
PM_tasks = {task:edges for (task, edges) in zip(PM_tasks_list, (eval(edge) for edge in Throat_edges))}
Ipm = {task:starts for (task, starts) in zip(PM_tasks_list, Ipm_list)}
Fpm = {task:ends for (task, ends) in zip(PM_tasks_list, Fpm_list)}
PM_edges = [eval(edge) for edge in Throat_edges]


# In[37]:


#Corrective maintenance tasks along station throat

CMTdata1 = pd.read_excel('Maintenance tasks.xlsx', 'CMT(throat)')

CM_tasks_list = [CMTdata1['Maintenance task'][i] for i in range (0, len(CMTdata1['Maintenance task']))]
Platform_edges = [CMTdata1['Maintenance edges'][i] for i in range (0, len(CMTdata1['Maintenance task']))]
Duration1 = [CMTdata1['Duration (min)'][i] for i in range (0, len(CMTdata1['Maintenance task']))]
CM_tasks = {task:edges for (task, edges) in zip(CM_tasks_list, (eval(edge) for edge in Platform_edges))}
CM_edges = [eval(edge) for edge in Platform_edges]
Tm = {task: duration for (task, duration) in zip(CM_tasks_list, Duration1)}


# In[38]:


B


# In[39]:


#Corrective maintenance tasks along platform track

CMTdata2 = pd.read_excel('Maintenance tasks.xlsx', 'CMT(platform)')

CM_tasks_list = [CMTdata2['Maintenance task'][i] for i in range (0, len(CMTdata2['Maintenance task']))]
Duration2 = [CMTdata2['Duration (min)'][i] for i in range (0, len(CMTdata2['Maintenance task']))]
Platform_track = [CMTdata2['Platform track'][i] for i in range (0, len(CMTdata2['Maintenance task']))]
CM_platform = {platform: duration for (platform, duration) in zip(Platform_track, Duration2)}
for key in CM_platform:
    train_home.update({key: "CMtask"})
    train_exit.update({key: "CMtask"})
    At.update({key: tw1})
    rt.update({key:CM_platform[key]})
    Op.update({key:key})
    wt.update({key: 0})
    dwell_time.update({key:CM_platform[key]})
    Ip.append(key)
    I.append(key)
    B.append(key)


# In[40]:


def first_node(route):
    for edge in route:
        for node in edge:
            first = node[0]
            break
        break
    return(first_node)

def train_inroutes(train, platform):
    candidate_routes = []
    for route in R_in:
        first = list(route[0])[0]
        for edge in route:
            last = edge[-1]
            if first == train_home[train] and last == platform:
                candidate_routes.append(route)
    return(candidate_routes)

def train_outroutes(train, platform):
    candidate_routes = []
    for route in R_out:
        first = list(route[0])[0]
        for edge in route:
            last = edge[-1]
            if first == platform and last == train_exit[train]:
                candidate_routes.append(route)
    return(candidate_routes)

def not_train_outroutes(train, platform):
    candidate_routes = []
    for route in R_out:
        if route not in train_outroutes(train, platform):
                candidate_routes.append(route)
    return(candidate_routes)

def not_train_inroutes(train, platform):
    candidate_routes = []
    for route in R_in:
        if route not in train_inroutes(train, platform):
                candidate_routes.append(route)
    return(candidate_routes)


# In[41]:


#set of train routes affected by corrective maintenance tasks
Rm_in = []
for route in R_in:
    for edge in route:
        if edge in CM_edges:
            Rm_in.append(route)
Rm_out = []
for route in R_out:
    for edge in route:
        if edge in CM_edges:
            Rm_out.append(route)
    
#Duration of corrective maintenance tasks            
def get_CMkey(edge):
    for key in CM_tasks.keys():
        if str(edge) in CM_tasks[key]:
            return(key)
def Mtime(route):
    Required_time = 0
    for edge in route:
        if edge in CM_edges:
            CM_label = get_CMkey(edge)
            Required_time = Tm[CM_label]
            return(Required_time)
    return(Required_time)        


# In[42]:


def get_PMkey(edge):
    for key in PM_tasks.keys():
        if str(edge) in PM_tasks[key]:
            return(key)
        
#PM task start time
def start(edge):
    start_time = Ipm[get_PMkey(edge)]
    return(start_time)

#PM task end time
def end(edge):
    end_time = Fpm[get_PMkey(edge)]
    return(end_time)


#C_in value
def C_in(train, inroute):
    for edge in inroute:
        if edge in PM_edges and start(edge) <= At[train] + wt[train] <= end(edge):
            Cim = 1
            break
        else:
            Cim = 0
    return(Cim)
    
#C_out value
def C_out(train, outroute):
    for edge in outroute:
        if edge in PM_edges and start(edge) <= At[train] + wt[train] + dwell_time[train] + Hsp <= end(edge):
            Cim = 1
            break
        else:
            Cim = 0
    return(Cim)
    
#Cipm value
def Cipm(train, platform):
    if platform in PM_edges and start(platform) <= At[train] + wt[train] <= end(platform):
        Cipm = 1
        return(Cipm)
    else:
        Cipm = 0
        return(Cipm)


# In[43]:


#Crc value
def Crc(train):
    if train in Ir:
        Crc = 10
        return(Crc)
    else:
        Crc = 0
        return(Crc)
    
#Cost of reassigning train i to platform p
Cip = pd.read_excel('TPP_data.xls', 'PR costs')
Cip.set_index('Original Assignment', inplace=True)


# In[44]:


#Sets of train pairs used to filter conflict detection and resolution between trains 
X = [(i,j) for i in I for j in I if i != j and At[j]>=At[i]+wt[i]+dwell_time[i]+Hsp]

Y = [(i,j) for i in I for j in I if i !=j and (At[i]+wt[i])<At[j]<(At[i]+wt[i]+dwell_time[i]+Hsp)]

Z = [(i,j) for i in I for j in I if i!=j and (At[i]-wt[j])<=At[j]<=(At[i]+wt[i])]

V = [(i,j) for i in I for j in I if i!=j and train_home[i] == train_home[j] and At[j]>At[i]]


# In[45]:


from docplex.mp.model import Model

#We name the model developed Model TM ('TM' for train and maintenance)
mdl = Model("Model_TM")


# In[46]:


from docplex.mp.environment import Environment
env = Environment()
env.print_information()


# In[47]:


#Decision variables in the optimization model

#1. Arrival route variables
Xin_vars = mdl.binary_var_matrix(I, R_in, 'Inroute')

#2. Departure route variables
Xout_vars = mdl.binary_var_matrix(I, R_out, 'Outroute')

#3. Home-Platform variables
P_vars = mdl.binary_var_matrix(I, A, 'PlatformAssigned')


#4. Train's precedence on arrival at home signal
Xij_vars = mdl.binary_var_matrix(I, I, 'InboundsPrecedence')

#5. Shared platform by trains
Pij_vars = mdl.binary_var_matrix(I, I, 'SharedPlatform')


#6. departure time from home signal variable
Dt_vars = mdl.continuous_var_dict(I, lb = tw1, name = 'DepartureTime([%s])')


# In[48]:


#Objective function of the model
#without Crc
mdl.minimize(1*mdl.sum(Dt_vars[train] - At[train]for train in I) + 1*mdl.sum(P_vars[i, p] for i in I for p in A if p == 'Dummy') + 1*mdl.sum(Cip[Op[train]][platform]*P_vars[train, platform] for train in I for platform in A if platform != 'Dummy') + 1*mdl.sum(C_in(train, route)*Xin_vars[train, route] for train in I for route in R_in) + 1*mdl.sum(C_out(train, route)*Xout_vars[train, route] for train in I for route in R_out) + 1*mdl.sum(Cipm(train, platform)*P_vars[train, platform] for train in I for platform in A))


# In[49]:


mdl.minimize(1*mdl.sum(Dt_vars[train] - At[train]for train in I) + 1*mdl.sum(P_vars[i, p] for i in I for p in A if p == 'Dummy') + 1*mdl.sum((Cip[Op[train]][platform]+Crc(train))*P_vars[train, platform] for train in I for platform in A if platform != 'Dummy') + 1*mdl.sum(C_in(train, route)*Xin_vars[train, route] for train in I for route in R_in) + 1*mdl.sum(C_out(train, route)*Xout_vars[train, route] for train in I for route in R_out) + 1*mdl.sum(Cipm(train, platform)*P_vars[train, platform] for train in I for platform in A))


# In[50]:


#This constraint ensures only one platform is assigned to each train 
for train in I:
    mdl.add_constraint(mdl.sum(P_vars[train, platform] for platform in A) == 1)


# In[51]:


#This constraint ensures that trains already dwelling at platforms are assigned their dwelling platforms
for train in I:
    if train in Ip:
        platform = Op[train]
        mdl.add_constraint(P_vars[train, platform] == 1)


# In[52]:


#This constraint ensures that if two trains are assigned same platform, the variable Pij equals 1
for i in I:
    for j in I:
        for platform in A:
            if i!=j:
                mdl.add_if_then(P_vars[i, platform] + P_vars[j, platform] == 2, Pij_vars[i,j] == 1)


# In[53]:


#This constraint ensures that if two trains are assigned different platforms, the variable Pij equals 0
for i in I:
    for j in I:
        for platform in A:
            if i!=j:
                mdl.add_if_then(P_vars[i, platform] + P_vars[j, platform] == 1, Pij_vars[i,j] == 0)


# In[54]:


#If train pair(i, j) are at same or different platforms, then train pair (j, i)
#are also at same or different platform
mdl.add_constraints(Pij_vars[i,j] == Pij_vars[j,i] for i in I for j in I if i != j)


# In[55]:


#This constraint ensures that an inbound train is assigned a route that connects it to its assigned platform
mdl.add_constraints(mdl.sum(Xin_vars[i,r] for r in train_inroutes(i,p)) == P_vars[i,p] for i in I for p in A if train_home[i] != 'CMtask')


# In[56]:


#This ensures that an outbound train is assigned a route that connects it to its departure direction
#from its assigned platform
mdl.add_constraints(mdl.sum(Xout_vars[i,r] for r in train_outroutes(i,p)) == P_vars[i,p] for i in I for p in A if train_home[i] != 'CMtask')


# In[57]:


#This ensures that each train is assigned only one inbound route and only one outbound route
for train in I:
    if train_home[i] != 'CMtask':
        mdl.add_constraint(mdl.sum(Xin_vars[train, route] for route in R_in) ==1)
        mdl.add_constraint(mdl.sum(Xout_vars[train, route] for route in R_out) ==1)


# In[58]:


#This ensures that trains already in the station (at tw1) are not queued at the home signal
mdl.add_constraints(Dt_vars[train] == tw1 for train in Ip)


# In[60]:


#This ensures that trains coming to the station are not assigned platforms currently occupied by trains in set Ip
mdl.add_constraints(Dt_vars[train] >= rt[platform]*P_vars[train, platform] for train in Ih for platform in B)


# In[61]:


#Ensures that departure time of all trains is greater than or equal to their arrival times
mdl.add_constraints(Dt_vars[train] >= At[train] for train in I)


# In[62]:


#This introduces a conflict filter by exempting these train pairs from
#departure precedence check by the optimization model
for pair in V + X + Y:
    i = pair[0]
    j = pair[1]
    mdl.add_constraint(Xij_vars[i,j] == 1)


# In[63]:


#This ensures that for every train pair, exactly one train must depart before the other.
for i in I:
    for j in I:
        if i != j:
            mdl.add_constraint(Xij_vars[i, j] + Xij_vars[j, i] == 1)
            


# In[64]:


#This ensures that departure from home signal and platform assignment are conflict-free
for pair in Y+Z:
    i = pair[0]
    j = pair[1]
    if j not in Ip:
        mdl.add_constraint(Dt_vars[j] >= Dt_vars[i] + (Hsp + dwell_time[i])*Pij_vars[i,j] - M*Xij_vars[j,i])


# In[65]:


#This ensures that a train is not delayed at home signal beyond its maximum allowable waiting time. 
mdl.add_constraints(Dt_vars[train] <= At[train] + wt[train]  for train in I)


# In[66]:


#This ensures that a train does not use a route under maintenance while coming into the station from home signal
mdl.add_constraints(Dt_vars[train] <= At[train] + wt[train] + Xin_vars[train, route]*Mtime(route) for train in I for route in R_in if train_home[i] != 'CMtask')


# In[67]:


#This ensure that a train does not use a route under maintenance while departing the station from platform
mdl.add_constraints(Dt_vars[train] + Hsp + dwell_time[train] >= Xout_vars[train, route]*Mtime(route) for train in I for route in R_out if train_home[i] != 'CMtask')


# In[68]:


solution = mdl.solve(log_output=True)


# In[69]:


mdl.print_information()


# In[70]:


print(solution)


# In[71]:


solve_details = mdl.get_solve_details()
print(solve_details)


# In[ ]:




