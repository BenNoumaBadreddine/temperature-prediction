import datetime
import pickle

from sklearn.linear_model import LinearRegression
from data_preparation.data_load import get_data_per_equipment

equipment_id = '8033'
train_data, test_data = get_data_per_equipment(equipment_id)
features = ['main_thermocouple_reading', 'average_temperature_zones', 'average_power', 'total_flow_reading',
            'load_temperature', 'Cooling_Motor', 'cooler']
target = ['main_thermocouple_reading_at_10_minute_ahead']
train_x = train_data[features].values
train_y = train_data[target].values
test_x = test_data[features].values
test_y = test_data[target].values

# Build and train the model
reg = LinearRegression()
reg = reg.fit(train_x, train_y)
print("The score on the train is:", reg.score(train_x, train_y))
print("The score on the test is:", reg.score(test_x, test_y))
print("The coefficients are: ", reg.coef_)

# dump the model
filename = "../models/" + equipment_id + "_best_model_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + \
           ".pickle"
pickle.dump(reg, open(filename, 'wb'))
