import pyomo.environ as pyEnv

file = open('states_line.csv')
lines = file.readlines()
file.close()

cost_matrix = []

for i in range(len(lines)):
    aux = lines[i][:-1].split(',')
    aux = [float(i) for i in aux if i != '']
    cost_matrix.append(aux)

n = len(cost_matrix)

# Model
model = pyEnv.ConcreteModel()

# Indexes for the cities
model.M = pyEnv.RangeSet(n)
model.N = pyEnv.RangeSet(n)

# Index for the dummy variable u
model.U = pyEnv.RangeSet(2, n)

# Decision variables xij
model.x = pyEnv.Var(model.N, model.M, within=pyEnv.Binary)

# Dummy variable ui
model.u = pyEnv.Var(model.N, within=pyEnv.NonNegativeIntegers, bounds=(0, n-1))

# Cost Matrix cij
model.c = pyEnv.Param(
    model.N, model.M, initialize=lambda model, i, j: cost_matrix[i-1][j-1])


def func_obj(model):
    return sum(model.x[i, j] * model.c[i, j] for i in model.N for j in model.M)


model.objective = pyEnv.Objective(rule=func_obj, sense=pyEnv.minimize)


def restricao_tipo_1(model, M):
    return sum(model.x[i, M] for i in model.N if i != M) == 1


model.const1 = pyEnv.Constraint(model.M, rule=restricao_tipo_1)


def restricao_tipo_2(model, N):
    return sum(model.x[N, j] for j in model.M if j != N) == 1


model.rest2 = pyEnv.Constraint(model.N, rule=restricao_tipo_2)


def restricao_tipo_3(model, i, j):
    if i != j:
        return model.u[i] - model.u[j] + model.x[i, j] * n <= n-1
    else:
        # Yeah, this else doesn't say anything
        return model.u[i] - model.u[i] == 0


model.rest3 = pyEnv.Constraint(model.U, model.N, rule=restricao_tipo_3)

solver = pyEnv.SolverFactory('cplex')
result = solver.solve(model, tee=False)

print(result)

l = list(model.x.keys())
for i in l:
    if model.x[i]() != 0:
        print(i, '--', model.x[i]())
