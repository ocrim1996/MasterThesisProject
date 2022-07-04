from gurobipy import *
import numpy as np
import os

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

time_to_opt = 0
tot_time = 0


class StQPSolverEfficientGurobi(object):
    def __init__(self, Q, c):
        assert(np.shape(Q)[0] == np.shape(Q)[1] and np.shape(c)[0] == np.shape(Q)[0])
        # Controlla se la matrice è simmetrica
        assert(np.all(Q-Q.T < 1e-7))
        self.Q = Q
        self.c = c
        self.n = np.shape(c)[0]

    def solve(self):
        # Creating Gurobi model, set debug to True to print debug mode.
        model = Model()
        debug = True
        gurobi_limit = True
        time_limit = 3600
        gurobi_time_limit = time_limit
        time_limit = time_limit

        # print(jacobian)

        if not debug:
            # Quieting Gurobi output
            model.setParam("OutputFlag", False)
        if gurobi_limit:
            model.setParam("TimeLimit", gurobi_time_limit)
        else:
            model.setParam("TimeLimit", time_limit)
        # model.setParam("Threads", n_cpus)
        model.setParam("IntFeasTol", 1e-09)
        model.setParam("Threads", 1)
        model.params.NonConvex = 2

        # Add variables to the model
        x = []
        z = []
        n = self.n

        # Inizializzo il vettore x con vincoli 0 < x < 1.
        for j in range(n):
            x.append(model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x{}".format(j)))

        # Inizializzo la matrice z che è uguale x^Tx, che ci serve poi nella funzione obiettivo xTQx + c*x.
        for j in range(n):
            z.append([])
            for i in range(n):
                z[j].append(model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="z{}_{}".format(j, i)))
                model.addConstr(z[j][i] == x[i]*x[j])

        #TODO Carefully check if factor 0.5 is present in front of the quadratic term
        f = quicksum(z[i][j]*self.Q[i, j] for i in range(n) for j in range(n)) + quicksum(x[i]*self.c[i] for i in range(n))

        model.setObjective(f)

        model.addConstr(quicksum(x[i] for i in range(n)) <= 1)
        model.addConstr(quicksum(x[i] for i in range(n)) >= 1)

        # Solve
        model.optimize(time_callback)

        if debug:
            for v in model.getVars():
                print("Var: {}, Value: {}".format(v.varName, v.x))

        global tot_time
        tot_time = model.Runtime
        return [model.getVarByName("x{}".format(j)).x for j in range(n)],  model.getObjective().getValue()


def time_callback(model, where):
    global time_to_opt
    if where == GRB.Callback.MIPSOL:
        time_to_opt = model.cbGet(GRB.Callback.RUNTIME)
        # best = model.cbGet(GRB.Callback.MIP_OBJBST)