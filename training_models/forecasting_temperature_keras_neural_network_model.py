from nitride_database_parser import get_job_date_time
from temperature_forecasting.inference.testing_temperature_forecasting_model import predict_temperature
from temperature_forecasting.keras_regression_model.keras_regression_functions import \
    mini_batch_gradient_descent_learning_algorithm
import datetime
from temperature_forecasting.data_preparation.data_preprocessing import get_readings_targets_columns, \
    create_job_timestamps
from temperature_forecasting.data_preparation.furnace_data_load import furnace_data_load, get_furnace_name_id
from temperature_forecasting.data_preparation.furnace_training_testing_data_preparation import furnace_training_testing_data_preparation, \
    get_training_testing_data

# choose the furnace
from temperature_forecasting.plotting_tool.plot_forecasting_results import plot_temperature_forecasting_of_job

system_identifier_id = "73b7652c-cf54-11eb-a074-02e1e5732d58"  # Chicago 8033
system_identifier_id = "6c1823c2-cf10-11eb-8ac4-02e1e5732d58"  # Chicago 8026

furnace_name = get_furnace_name_id(system_identifier_id)
dataframes_per_furnace_dict = furnace_data_load(system_identifier_id)
# dataframes_per_furnace_dict = dataframes_per_furnace_dict['dataframes_per_furnace_dict']


for key, data in dataframes_per_furnace_dict.items():
    data['average_temperature_zones'] = data['sum_temperature_zones']/6
    job_date_time = get_job_date_time(data['job_id'].iat[0])
    data['job_date_time'] = job_date_time
    job_timestamps = create_job_timestamps(data)
    data['job_timestamps'] = job_timestamps

    dataframes_per_furnace_dict.update({key: data})

truncated_furnace_train_dict, truncated_furnace_test_dict, series_train, series_test = \
    furnace_training_testing_data_preparation(system_identifier_id, dataframes_per_furnace_dict)

main_thermocouple_reading, zones_temperature, zones_power, target_main_thermocouple, zone_temperature_target = \
    get_readings_targets_columns(list(series_train.columns))

temperature_features = main_thermocouple_reading + zones_temperature + zones_power + \
                       ['total_flow_reading', 'load_temperature', 'cooler'] + target_main_thermocouple

# UNIVERSAL INPUT VECTOR
temperature_features = main_thermocouple_reading + \
                       ['average_temperature_zones', 'average_power', 'total_flow_reading', 'load_temperature',
                        'Cooling_Motor', 'cooler'] + target_main_thermocouple

targets = target_main_thermocouple
X_train, Y_train, X_test, Y_test = get_training_testing_data(series_train, series_test, temperature_features, targets)

# keras model
checkpoint_file_path = "C:/NitrexGroupGit/machine-learning/temperature_forecasting/models/" + furnace_name \
                       + "_best_model_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"

PARAMS = {'features': temperature_features,
          'batch_size': 64,
          'n_epochs': 10,
          'activation': 'relu',
          'dense_units': 256,
          'optimizer': 'adam',
          'loss_metrics': 'mean_squared_error',
          'input_dim': len(temperature_features) - len(targets),
          'learning_rate': 0.001,
          'patience': 10,
          'validation_split': 0.2,
          'output_dim': 1
          }
estimator, history = mini_batch_gradient_descent_learning_algorithm(X_train, Y_train, param=PARAMS,
                                                                    checkpoint_file_path=checkpoint_file_path)

truncated_furnace_test_dict = predict_temperature(estimator, truncated_furnace_test_dict, X_test,
                                                  zone="main_thermocouple_reading", sklearn_model=False)

plot_temperature_forecasting_of_job(truncated_furnace_test_dict, job_nbr=55, zone="main_thermocouple_reading")

# path_furnace_name = furnace_name.replace("_", "/")+"/Log/"
# plot_prediction_vs_real_time_values(series_test, path_furnace_name, truncated_furnace_test_dict,
#                                     zone=main_thermocouple_reading)
