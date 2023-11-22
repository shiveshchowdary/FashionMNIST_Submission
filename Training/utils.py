import tensorflow as tf
from tensorflow.keras import layers, models, Model
from itertools import product
from tensorflow.keras.callbacks import EarlyStopping

# from tqdm import tqdm

class ModelCNN:
    def __init__(self):
        pass
    def create_model_cnn(self, filters_conv1, filters_conv2, filter_size, dense_layer_inp, input_shape):
        model = models.Sequential()
        model.add(layers.Conv2D(filters_conv1, (filter_size, filter_size),padding='same', activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(filters_conv2, (filter_size, filter_size),padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(dense_layer_inp, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        return model

    def grid_search_cnn(self, train_data, train_labels, val_data, val_labels, parameter_grid, epochs=1, batch_size=32):
        best_accuracy = 0
        best_parameters = None
        all_results = []

        for parameter_combination in product(*parameter_grid.values()):
            current_parameters = dict(zip(parameter_grid.keys(), parameter_combination))

            filters_conv1 = current_parameters["convolution1 filters"]
            filters_conv2 = current_parameters["convolution2 filters"]
            filter_size = current_parameters["filter_size"]
            dense_layer_inp = current_parameters["dense_layer"]
            input_shape = (28, 28, 1)
            model = self.create_model_cnn(filters_conv1, filters_conv2, filter_size, dense_layer_inp, input_shape)

            model.compile(optimizer='adam',
                          loss='SparseCategoricalCrossentropy',
                          metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            history = model.fit(train_data, train_labels,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(val_data, val_labels),
                                callbacks=[early_stopping],
                                verbose=0)

            _, current_accuracy = model.evaluate(val_data, val_labels)

            result = {'parameters': current_parameters, 'accuracy': current_accuracy, 'history': history}
            all_results.append(result)

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_parameters = current_parameters

        return best_parameters, best_accuracy, all_results

class ModelResNet:
    def __init__(self):
        pass
    def create_model_resnet(self, input_shape, num_classes, num_residual_blocks=3, num_filters=32):
        input_tensor = tf.keras.Input(shape=input_shape)

        x = layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(input_tensor)

        for _ in range(num_residual_blocks):
            residual = x
            x = layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(x)
            x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
            x = layers.Add()([x, residual])
            x = layers.Activation('relu')(x)

        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(64, activation='relu')(x)

        output_tensor = layers.Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=input_tensor, outputs=output_tensor)

        return model
    def grid_search_resnet(self, train_data, train_labels, val_data, val_labels, parameter_grid, epochs=1, batch_size=32):
        best_accuracy = 0
        best_parameters = None
        all_results = []

        for parameter_combination in product(*parameter_grid.values()):
            current_parameters = dict(zip(parameter_grid.keys(), parameter_combination))

            num_res = current_parameters["residual blocks"]
            filter_size = current_parameters["filters"]
            input_shape = (28, 28, 1)
            model = self.create_model_resnet(input_shape, 10, num_res, filter_size)

            model.compile(optimizer='adam',
                          loss='SparseCategoricalCrossentropy',
                          metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            history = model.fit(train_data, train_labels,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(val_data, val_labels),
                                callbacks=[early_stopping],
                                verbose=0)

            _, current_accuracy = model.evaluate(val_data, val_labels)

            result = {'parameters': current_parameters, 'accuracy': current_accuracy, 'history': history}
            all_results.append(result)

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_parameters = current_parameters

        return best_parameters, best_accuracy, all_results



