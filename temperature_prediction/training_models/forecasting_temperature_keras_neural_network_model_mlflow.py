import datetime
import mlflow
import os
from temperature_prediction.data_preparation.data_load import get_data_per_equipment
from temperature_prediction.keras_regression_model.keras_regression_functions import mini_batch_gradient_descent_learning_algorithm

equipment_id = '8033'
train_data, test_data = get_data_per_equipment(equipment_id)
features = ['main_thermocouple_reading', 'average_temperature_zones', 'average_power', 'total_flow_reading',
            'load_temperature', 'Cooling_Motor', 'cooler']
target = ['main_thermocouple_reading_at_10_minute_ahead']

train_x = train_data[features].values
train_y = train_data[target].values
test_x = test_data[features].values
test_y = test_data[target].values

# keras model
checkpoint_file_path = "../models/" + equipment_id + "_best_model_" + \
                       datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"

PARAMS = {'features': features,
          'batch_size': 64,
          'n_epochs': 1,
          'activation': 'relu',
          'dense_units': 256,
          'optimizer': 'adam',
          'loss_metrics': 'mean_squared_error',
          'input_dim': len(features),
          'learning_rate': 0.001,
          'patience': 10,
          'validation_split': 0.2,
          'output_dim': 1
          }


from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient


def set_experiment(experiment_name: str, artifact_location: str):
    """
    Set given experiment as active experiment. If experiment does not exist, create an experiment
    with provided name.
    :param experiment_name: Name of experiment to be activated.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    exp_id = experiment.experiment_id if experiment else None
    if exp_id is None:  # id can be 0
        print("INFO: '{}' does not exist. Creating a new experiment".format(experiment_name))
        exp_id = client.create_experiment(experiment_name, artifact_location=artifact_location)
    elif experiment.lifecycle_stage == LifecycleStage.DELETED:
        raise MlflowException(
            "Cannot set a deleted experiment '%s' as the active experiment."
            " You can restore the experiment, or permanently delete the "
            " experiment to create a new one." % experiment.name)
    global _active_experiment_id
    _active_experiment_id = exp_id


experiment_name = 'temperature_prediction_5'
artifact_location = os.environ.get('MLFLOW_ARTIFACT_LOCATION')
tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri(tracking_uri)
set_experiment(experiment_name, artifact_location)
experiment = mlflow.get_experiment(_active_experiment_id)


print("Experiment Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

mlflow.set_experiment(experiment_name)

# Auto log all the parameters, metrics, and artifacts
mlflow.tensorflow.autolog()


tags = {"engineering": "ML Platform",
        "engineering_remote": "ML Platform"}
description = 'This is a project of predicting temperature during a process'
with mlflow.start_run(run_name="Keras neural network", description=description):
    estimator, history = mini_batch_gradient_descent_learning_algorithm(train_x, train_y, param=PARAMS,
                                                                        checkpoint_file_path=checkpoint_file_path)
    from sklearn import linear_model

    reg = linear_model.LinearRegression()
    reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    mlflow.sklearn.log_model(reg, "model")


    # Log mlflow attributes for mlflow UI
    mlflow.log_param("PARAMS", PARAMS)
    mlflow.sklearn.log_model(estimator, "model")
    # mlflow.set_tag("mlflow.user", "NAME")
    mlflow.set_tags(tags)

    #
    # truncated_furnace_test_dict = predict_temperature(estimator, truncated_furnace_test_dict, test_x,
    #                                                   zone="main_thermocouple_reading", sklearn_model=False)
    #
    # plot_temperature_forecasting_of_job(truncated_furnace_test_dict, job_nbr=55, zone="main_thermocouple_reading")

    # path_furnace_name = furnace_name.replace("_", "/")+"/Log/"
    # plot_prediction_vs_real_time_values(series_test, path_furnace_name, truncated_furnace_test_dict,
    #                                     zone=main_thermocouple_reading)
