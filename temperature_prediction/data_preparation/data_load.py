import os
import pandas as pd


def get_data_per_equipment(equipment_id: str):
    """
    Returns A dictionary of DataFrames having a set of calculated features useful for a later temperature forecasting.
    :param: EQUIPMENT_ID: The id of the equipment.
    :return: train_data, test_data: two dataframes.
    """
    train_exists = os.path.isfile('../data/' + equipment_id + '_train_data.csv')
    test_exists = os.path.isfile('../data/' + equipment_id + '_test_data.csv')
    if train_exists and test_exists:
        train_data = pd.read_csv('../data/' + equipment_id + '_train_data.csv')
        test_data = pd.read_csv('../data/' + equipment_id + '_test_data.csv')
        return train_data, test_data
    else:
        raise Exception(f'The given data directory has no data. Please check again the existence of the data.')
