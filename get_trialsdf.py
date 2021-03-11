import optuna
import pandas

study_name = 'example-study'  # Unique identifier of the study.
study = optuna.create_study(study_name=study_name, storage='sqlite:///example.db', load_if_exists=True)
#extract the trials into a pandas dataframe and convert to csv
df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
df.to_csv('trials.csv')
