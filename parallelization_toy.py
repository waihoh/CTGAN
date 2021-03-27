import optuna

#create a study and store in a db file
study_name = 'example-study'  # Unique identifier of the study.
study = optuna.create_study(study_name=study_name, storage="mysql://root:root@localhost:8888/ml_expts", load_if_exists=True)

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2

#run one study
study.optimize(objective, n_trials=3)

#1. run parallelization_toy.py in multiple terminals
#2 extract the dataframe trials in get_trialsdf.py

