import pygad
import numpy as np
import pyomo
from pyomo.environ import ConcreteModel, Var, Objective, value

# Define the Pyomo model
def create_pyomo_model(individual):
    #print(individual)
    model = ConcreteModel()
    model.x = Var(initialize=individual[0])
    model.y = Var(initialize=individual[1])
    
    # Example objective: minimize (x-3)^2 + (y-2)^2
    model.obj = Objective(expr=(model.x - 3)**2 + (model.y - 2)**2)
    
    return model

# Define the fitness function for PyGAD
def fitness_func(ga_instance, solution, solution_idx):
    #print(solution)
    model = create_pyomo_model(solution)
    fitness = value(model.obj)
    return -fitness  # PyGAD maximizes fitness, so we return the negative value to minimize

# Configure PyGAD parameters
ga_instance = pygad.GA(num_generations=40,
                       num_parents_mating=10,
                       fitness_func=fitness_func,
                       sol_per_pop=50,
                       num_genes=2,
                       init_range_low=-10,
                       init_range_high=10,
                       parent_selection_type="tournament",
                       keep_parents=2,
                       crossover_type="scattered",
                       mutation_type="random",
                       mutation_percent_genes=20)

# Run the genetic algorithm
ga_instance.run()

ga_instance.save_solutions = True
#print(ga_instance.solutions())
#print(f"Solutions: {solutions}")
#solutions = ga_instance.best_solutions()
#print(f"Best solutions: {solutions}")

# Extract the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best solution: {solution}, Fitness: {solution_fitness}")
