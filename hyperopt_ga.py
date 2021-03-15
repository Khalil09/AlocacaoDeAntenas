import pygad
import numpy
import parametros as p
import random
import pandas
import os
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, STATUS_OK, Trials, hp
from GA_lib import fitness, individual, create_population, callback_generation

space = {
    'crossover-probability': hp.uniform('crossover-probability', 0.1, 1),
    'mutation-probability': hp.uniform('mutation-probability', 0.1, 1),
    'elitism-rate': hp.uniform('elitism-rate', 0.1, 1),
    'crossover-type': hp.choice('crossover-type', ['single_point', 'two_points']),
    'selection-type': hp.choice('selection-type', ['rws', 'rank']),
    'keep-parents': hp.choice('keep-parents', list(range(1, 6))),
}

def objective(params):
    ga_instance = pygad.GA(num_generations=p.generations,
                        num_parents_mating=int(params['elitism-rate']*p.population_size),
                        fitness_func=fitness,
                        initial_population=create_population(),
                        num_genes=p.N,
                        parent_selection_type=params['selection-type'],
                        keep_parents=5,
                        crossover_type=params['crossover-type'],
                        crossover_probability=params['crossover-probability'],
                        mutation_type=p.mutation_type,
                        mutation_percent_genes=params['mutation-probability'],
                        on_generation=callback_generation)
    
    ga_instance.run()

    _, solution_fitness, _ = ga_instance.best_solution()

    return {'loss': -solution_fitness, 'status': STATUS_OK }

trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=10)
df = pandas.DataFrame()

trial_dict = {}
for t in trials.trials:
    trial_dict.update(t['misc']['vals'])
    trial_dict.update(t['result'])
    df = df.append(trial_dict, ignore_index=True)

print (best)
print (trials.best_trial)

outname = 'hyperopt_GA.csv'
outdir = 'results'
if not os.path.exists(outdir):
    os.mkdir(outdir)

fullname = os.path.join(outdir, outname)

df.to_csv(fullname, mode='a')

best_df = pandas.DataFrame()
best_df = best_df.append(trials.best_trial, ignore_index=True)

outname = 'best_trial.csv'
if not os.path.exists(outdir):
    os.mkdir(outdir)

fullname = os.path.join(outdir, outname)

best_df.to_csv(fullname, mode='a')