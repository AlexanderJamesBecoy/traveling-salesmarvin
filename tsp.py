import numpy as np
import pyomo.environ as pyEnv

"""
We define the cost of two locations as the distance between them. As we operate in the euclidian plane, the bee line can be used. 
For one cluster (TSP), including the depot, we store all costs in a so called cost matrix. 
"""
def distance(loc1, loc2):
    return np.sqrt(np.sum((loc1-loc2)**2))

def def_cost_matrix(locations):
    cost_matrix = np.zeros((len(locations), len(locations))) # creates matrix with all values zero
    # Here multiple lines of code need to be filled
    for i in range(len(locations)):
        for j in range(len(locations)):
            cost_matrix[i][j] = distance(locations[i], locations[j])

    return cost_matrix

"""
Define functions to model the objective function and the needed constraints
"""

def obj_func(model):
    return sum(model.x[i,j] * model.c[i,j] for i in model.N for j in model.M)

def rule_const1(model,M):
    return sum(model.x[i,M] for i in model.N if i!=M ) == 1

def rule_const2(model,N):
    return sum(model.x[N,j] for j in model.M if j!=N) == 1

def rule_const3(model,i,j):
    if i!=j:
        return model.u[i] - model.u[j] + model.x[i,j] * len(model.u) <= len(model.u)-1
    else:
        #Yeah, this else doesn't say anything
        return model.u[i] - model.u[i] == 0 

"""
The model needs to be created first. To solve it we will call a solver (glpk). 
Last, the output of the solver needs to be shaped (to further work with it) and returned.
"""

def solve_tsp(locations_tsp, cost_matrix): # based on "http://www.opl.ufc.br/post/tsp/"
    #Model
    model = pyEnv.ConcreteModel()

    #Indexes for the cities
    model.M = pyEnv.RangeSet(len(locations_tsp))                
    model.N = pyEnv.RangeSet(len(locations_tsp))

    #Index for the dummy variable u
    model.U = pyEnv.RangeSet(2,len(locations_tsp))
    
    #Decision variables xij
    model.x = pyEnv.Var(model.N,model.M, within=pyEnv.Binary)

    #Dummy variable ui
    model.u = pyEnv.Var(model.N, within=pyEnv.NonNegativeIntegers,bounds=(0,len(locations_tsp)-1))
    
    #Cost Matrix cij
    model.c = pyEnv.Param(model.N, model.M,initialize=lambda model, i, j: cost_matrix[i-1][j-1])

    model.objective = pyEnv.Objective(rule=obj_func,sense=pyEnv.minimize)

    model.const1 = pyEnv.Constraint(model.M,rule=rule_const1)

    model.rest2 = pyEnv.Constraint(model.N,rule=rule_const2)

    model.rest3 = pyEnv.Constraint(model.U,model.N,rule=rule_const3)

    #model.print()

    #Solves
    solver = pyEnv.SolverFactory('glpk')
    result = solver.solve(model,tee = False)

    #Prints the results
    #print(result)

    l = list(model.x.keys())
    sol=[]
    for i in l:
        if model.x[i]() != 0:
            if model.x[i]() != None:
                sol.append(i)
    print(f"Sol: {sol}")
    
    #sort the solution
    sorted_sol = [1] #Initalize a list of visited location ids, always starting at the depot
    for i in sol:
        for ii in sol:
            last_loc = sorted_sol[-1]
            if ii[0] == last_loc:
                if ii[1] == 1: # stop if we are back at the depot
                    break
                else:
                    sorted_sol.append(ii[1])
    sorted_sol.append(1) # we go back to the depot
            
    return sorted_sol