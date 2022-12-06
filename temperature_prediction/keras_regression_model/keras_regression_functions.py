from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import time
import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, LambdaCallback
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
import neptune

feature_list = []


def log_data(logs):
    # neptune.log_metric('Titre: epoch_accuracy', logs['accuracy'])
    neptune.log_metric('Train_loss', logs['loss'])
    # neptune.log_metric('Titre: epoch_val_accuracy', logs['val_accuracy'])
    neptune.log_metric('Val_loss', logs['val_loss'])


Lmd = LambdaCallback(on_epoch_end=lambda epoch, logs: log_data(logs))


def create_keras_model(learning_rate, activation, dense_units, input_dim, output_dim, loss_metrics):
    # Create model
    seed_initializer = 0
    model = Sequential()
    model.add(
        Dense(dense_units, input_dim=input_dim, kernel_initializer=initializers.glorot_uniform(seed=seed_initializer),
              activation=activation, activity_regularizer=regularizers.l2(1e-5)))
    # model.add(BatchNormalization()) model.add(Dropout(0.50)) The Glorot uniform or normal initializer, also called
    # Xavier uniform or normal initializer. it is the well-used initializer for the neural network weights
    model.add(
        Dense(dense_units, kernel_initializer=initializers.glorot_uniform(seed=seed_initializer), activation=activation,
              activity_regularizer=regularizers.l2(1e-5)))
    #     model.add(Dropout(0.50))
    model.add(
        Dense(dense_units, kernel_initializer=initializers.glorot_uniform(seed=seed_initializer), activation=activation,
              activity_regularizer=regularizers.l2(1e-5)))
    #     model.add(Dropout(0.50))
    model.add(Dense(output_dim, kernel_initializer=initializers.glorot_uniform(seed=seed_initializer)))
    # Compile model
    opt = Adam(learning_rate=learning_rate)
    model.compile(loss=loss_metrics, optimizer=opt)
    return model


def mini_batch_gradient_descent_learning_algorithm(normalized_X_train, desired_output_train, param: dict,
                                                   checkpoint_file_path: str = "models/best_model_" + datetime.datetime.now().strftime(
                                                       "%Y%m%d-%H%M%S") + ".h5"):
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_file_path, monitor='val_loss', mode='min',
                                       save_best_only=True, verbose=1)
    estimator = KerasRegressor(build_fn=create_keras_model, learning_rate=param['learning_rate'],
                               activation=param['activation'], dense_units=param['dense_units'],
                               input_dim=param['input_dim'], output_dim=param['output_dim']
                               , loss_metrics=param['loss_metrics'])

    start_time = time.time()
    early_stopping = EarlyStopping(monitor='val_loss', patience=param['patience'], restore_best_weights=True)
    history = estimator.fit(normalized_X_train, desired_output_train,
                            epochs=param['n_epochs'], batch_size=param['batch_size'],
                            validation_split=param['validation_split'], verbose=1,
                            shuffle=True, callbacks=[early_stopping, model_checkpoint])
    # log model summary with neptune
    # estimator.model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))
    return estimator, history


def mini_batch_gradient_descent_learning_algorithm_with_validation(normalized_X_train, desired_output_train,
                                                                   normalized_X_validation, desired_output_validation,
                                                                   param: dict, checkpoint_file_path:
        str = "models/best_model_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"):

    model_checkpoint = ModelCheckpoint(filepath=checkpoint_file_path, monitor='val_loss', mode='min',
                                       save_best_only=True, verbose=1)
    estimator = KerasRegressor(build_fn=create_keras_model, learning_rate=param['learning_rate'],
                               activation=param['activation'], dense_units=param['dense_units'],
                               input_dim=param['input_dim'], output_dim=param['output_dim']
                               , loss_metrics=param['loss_metrics'])
    start_time = time.time()

    history = estimator.fit(normalized_X_train, desired_output_train,
                            epochs=param['n_epochs'], batch_size=param['batch_size'],
                            validation_data=(normalized_X_validation, desired_output_validation),
                            verbose=1,
                            shuffle=True,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=param['patience'],
                                                     restore_best_weights=True), model_checkpoint, Lmd])
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))

    return estimator, history
