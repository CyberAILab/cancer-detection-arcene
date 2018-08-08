import os
import numpy as np

def read_dataset_item(filename, multidim_values = True):
    with open(filename) as f:
        if(multidim_values):
            array = [[int(x) for x in line.split()] for line in f]
        else:
            array = []
            for line in f:
                for x in line.split():
                    val = int(x)
                    if(val < 0):
                        array.append(0)
                    else:
                        array.append(1)
    return array



def read_dataset(folder=".\\ARCENE"):
    test_data_filename = os.path.join(folder, "arcene_test.data")
    train_data_filename = os.path.join(folder, "arcene_train.data")
    train_labels_filename = os.path.join(folder, "arcene_train.labels")
    validation_data_filename = os.path.join(folder, "arcene_valid.data")
    validation_labels_filename = os.path.join(folder, "arcene_valid.labels")

    test_data_array = np.array(read_dataset_item(test_data_filename))
    train_data_array = np.array(read_dataset_item(train_data_filename))
    train_label_array = np.array(read_dataset_item(train_labels_filename, False))
    validation_data_array = np.array(read_dataset_item(validation_data_filename))
    validation_label_array = np.array(read_dataset_item(validation_labels_filename, False))

    return test_data_array, train_data_array, train_label_array, validation_data_array, validation_label_array
