import datetime
import neptune
import os
from data_preparation.data_load import get_data_per_equipment
from keras_regression_model.keras_regression_functions import mini_batch_gradient_descent_learning_algorithm

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

NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
neptune.init(project_qualified_name="badreddine/sandbox", api_token=NEPTUNE_API_TOKEN)

with neptune.create_experiment(name='keras-integration-example', params=PARAMS,
                               tags=["temperature-forecasting", "keras-model", 'Neural-network']):

    estimator, history = mini_batch_gradient_descent_learning_algorithm(train_x, train_y, param=PARAMS,
                                                                        checkpoint_file_path=checkpoint_file_path)

    # save the best model
    neptune.log_artifact(checkpoint_file_path)
    print("Saved model to disk")


    #
    # truncated_furnace_test_dict = predict_temperature(estimator, truncated_furnace_test_dict, test_x,
    #                                                   zone="main_thermocouple_reading", sklearn_model=False)
    #
    # plot_temperature_forecasting_of_job(truncated_furnace_test_dict, job_nbr=55, zone="main_thermocouple_reading")

    # path_furnace_name = furnace_name.replace("_", "/")+"/Log/"
    # plot_prediction_vs_real_time_values(series_test, path_furnace_name, truncated_furnace_test_dict,
    #                                     zone=main_thermocouple_reading)

