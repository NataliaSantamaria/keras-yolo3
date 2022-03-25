# Visualize and analyze training trials

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

import sklearn

#Cambiar dependiendo del estudio que quiera
study_name = "third-study"  # Unique identifier of the study.
storage_name = "sqlite:///third-study.db"

# Load previous study
study = optuna.load_study(study_name=study_name, storage=storage_name, pruner=optuna.pruners.SuccessiveHalvingPruner())

# Print results
# Get the dictionary of the (best) parameter name and parameter values:
best_params = study.best_params
found_hw = best_params["hw"]
print("Found hw: {}".format(found_hw))

# Get the best observed value
print("Best value: {}".format(study.best_value))
# Get the best trial
print("Best trial: {}".format(study.best_trial))
# Get all trials:
print("All trials: {}".format(study.trials))
# Get the number of trials:
print("Number of trials: {}".format(len(study.trials)))

# Export to pandas dataframe
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
print(df)

# Visualizations
#Visualize the optimization history. See plot_optimization_history() for the details.
fig1 = plot_optimization_history(study)
fig1.show()
#Visualize the learning curves of the trials. See plot_intermediate_values() for the details.
#fig2 = plot_intermediate_values(study) #You need to set up the pruning feature to utilize `plot_intermediate_values()`
#fig2.show()
#Visualize the hyperparameters importance
fig3 = plot_param_importances(study)
fig3.show()
#Visualize high-dimensional parameter relationships. See plot_parallel_coordinate() for the details.
fig4 = plot_parallel_coordinate(study)
fig4.show()
#Select parameters to visualize.
fig5 = plot_parallel_coordinate(study, params=["batch_size", "dropout_rate"])
fig5.show()
#Visualize hyperparameter relationships. See plot_contour() for the details.
fig6 = plot_contour(study)
fig6.show()
#Select parameters to visualize.
fig7 = plot_contour(study, params=["batch_size", "dropout_rate"])
fig7.show()
#Visualize individual hyperparameters as slice plot. See plot_slice() for the details.
fig8 = plot_slice(study)
fig8.show()
#Select parameters to visualize.
fig9 = plot_slice(study, params=["batch_size", "dropout_rate"])
fig9.show()
#Learn which hyperparameters are affecting the trial duration with hyperparameter importance.
fig10 = optuna.visualization.plot_param_importances(
    study, target=lambda t: t.duration.total_seconds(), target_name="duration"
)
fig10.show()
#Visualize empirical distribution function. See plot_edf() for the details.
fig11 = plot_edf(study)
fig11.show()