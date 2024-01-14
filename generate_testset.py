"""Example experiment for the 2 mismatched segments dilation model."""
import glob
import json
import logging
import os, sys
import tensorflow as tf
import numpy as np



import sys
# add base path to sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# from task1_match_mismatch.models.dilated_convolutional_model import dilation_model

# from util.dataset_generator import DataGenerator, batch_equalizer_fn, create_tf_dataset
# from task1_match_mismatch.models.testModel import lstm_mel 
# from task1_match_mismatch.models.LSTM_model import lstm_mel
from experiment_models import eeg_mha_dc_speech_gru_dc_model, TransformerBlock # MHA+DC for EEG and GRU+DC for speech stimulus
from dataset_generator import DataGenerator, batch_equalizer_fn, create_tf_dataset


def evaluate_model(model, test_dict):
    """Evaluate a model.

    Parameters
    ----------
    model: tf.keras.Model
        Model to evaluate.
    test_dict: dict
        Mapping between a subject and a tf.data.Dataset containing the test
        set for the subject.

    Returns
    -------
    dict
        Mapping between a subject and the loss/evaluation score on the test set
    """
    evaluation = {}
    for subject, ds_test in test_dict.items():
        logging.info(f"Scores for subject {subject}:")
        results = model.evaluate(ds_test, verbose=2)
        metrics = model.metrics_names
        evaluation[subject] = dict(zip(metrics, results))
    return evaluation

def predict_model(model, test_dict):
    """Predict a model.

    Parameters
    ----------
    model: tf.keras.Model
        Model to evaluate.
    test_dict: dict
        Mapping between a subject and a tf.data.Dataset containing the test
        set for the subject.

    Returns
    -------
    dict
        Mapping between a subject and the loss/evaluation score on the test set
    """
    evaluation = {}
    for subject, ds_test in test_dict.items():
        logging.info(f"Scores for subject {subject}:")
        results = model.predict(ds_test, verbose=2)
        metrics = model.metrics_names
        evaluation[subject] = dict(zip(metrics, results))
    return evaluation


if __name__ == "__main__":
    # Parameters
    # Length of the decision window
    window_length_s = 5
    fs = 64

    window_length = window_length_s * fs  # 5 seconds
    # Hop length between two consecutive decision windows
    hop_length = 64

    epochs = 60
    patience = 10
    batch_size = 64
    number_mismatch = 4 # or 4

    training_log_filename = "training_log_{}_{}_MHAGRU.csv".format(number_mismatch, window_length_s)

    # Get the path to the config gile
    experiments_folder = os.path.dirname(__file__)
    task_folder = os.path.dirname(experiments_folder)
    # util_folder = os.path.join(os.path.dirname(task_folder), "util")
    # config_path = os.path.join(util_folder, 'config.json')

    # Load the config
    # with open(config_path) as fp:
    #     config = json.load(fp)

    # Provide the path of the dataset
    # which is split already to train, val, test
    test_folder = 'test_folder'
    data_folder = os.path.join(test_folder)
    eeg_folder = os.path.join(data_folder, "preprocessed_eeg")
    stimulus_folder = os.path.join(data_folder, "stimulus")

    # stimulus feature which will be used for training the model. Can be either 'envelope' ( dimension 1) or 'mel' (dimension 28)
    #stimulus_features = ["envelope"]
    #stimulus_dimension = 1

    # uncomment if you want to train with the mel spectrogram stimulus representation
    stimulus_features = ["mel"]
    stimulus_dimension = 10

    features = ["eeg"] + stimulus_features

    # Create a directory to store (intermediate) results
    results_folder = os.path.join(experiments_folder, "results_dilated_convolutional_model_{}_MM_{}_s_{}".format(number_mismatch, window_length_s, stimulus_features[0]))
    os.makedirs(results_folder, exist_ok=True)

    # create dilation model
    # model = dilation_model(time_window=window_length, eeg_input_dimension=64, env_input_dimension=stimulus_dimension, num_mismatched_segments = number_mismatch)
    # model = eeg_mha_dc_speech_gru_dc_model([window_length, 64], [window_length, stimulus_dimension])
    model = eeg_mha_dc_speech_gru_dc_model(time_window=window_length, eeg_input_dimension=64, env_input_dimension=stimulus_dimension, num_mismatched_segments = number_mismatch) # MHA+DC for EEG and GRU+DC for speech stimulus
 
    model_path = os.path.join(results_folder, "model_{}_MM_{}_s_{}.h5".format(number_mismatch, window_length_s, stimulus_features[0]))
    print(model_path)
    model.load_weights(model_path)
    results_filename = 'testset_{}_{}_s.json'.format(number_mismatch, window_length_s)
    eeg_files = [x for x in glob.glob(os.path.join(eeg_folder, "*"))]
    stimulus_files = [x for x in glob.glob(os.path.join(stimulus_folder, "*"))]
    mapping_files = [x for x in glob.glob(os.path.join(data_folder, "sub-*"))]

    # print(eeg_folder, stimulus_folder)
    # print(eeg_files)
    # print(stimulus_files)

    eeg_dict = {}
    stimulus_dict = {}
    for npz in eeg_files:
        data = np.load(npz)
        for file in data:
            eeg_dict[file] = data[file]

    for npz in stimulus_files:
        data = np.load(npz)
        for file in data:
            stimulus_dict[file] = data[file]

    output = {}
    count = 0
    print(mapping_files)
    for file in mapping_files:
        mapping = json.load(open(file))
        for key, testcase in mapping.items():
            tmp = []
            # tmp.append(tf.convert_to_tensor(eeg_dict[testcase["eeg"]], dtype='float32'))
            tmp.append(np.expand_dims(eeg_dict[testcase["eeg"]], axis = 0))
            for sti in testcase["stimulus"]:
                # tmp.append(tf.convert_to_tensor(stimulus_dict[sti], dtype='float32'))
                tmp.append(np.expand_dims(stimulus_dict[sti], axis=0))
            
            # datasets_test[sub] = create_tf_dataset(test_generator, window_length, batch_equalizer_fn,
            #                                         hop_length, batch_size=1,
            #                                         number_mismatch=number_mismatch,
            #                                         data_types=(tf.float32, tf.float32),
            #                                         feature_dims=(64, stimulus_dimension))
            
            #print("tmp", tmp[1].shape)
            # tmp = tf.convert_to_tensor(tmp)
            output[key] = int(np.argmax(model.predict(tmp)[0]))
            count += 1
            #print(output[key])
        

    results_path = os.path.join(results_folder, "output1.json")
    with open(results_path, "w") as fp:
        json.dump(output, fp)
    logging.info(f"Results saved at {results_path}")
    




    # test_window_lengths = [5] #[3,5]
    # number_mismatch_test = [4] #[2,3,4, 8]
    # for number_mismatch in number_mismatch_test:
    #     for window_length_s in test_window_lengths:
    #         window_length = window_length_s * fs
    #         results_filename = 'testset_{}_{}_s.json'.format(number_mismatch, window_length_s)
    #         # Evaluate the model on test set
    #         # Create a dataset generator for each test subject
    #         test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if
    #                     os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    #         # Get all different subjects from the test set
    #         subjects = list(set([os.path.basename(x).split("_-_")[1] for x in test_files]))
    #         print(test_files)
    #         print(subjects)
    #         datasets_test = {}
    #         # Create a generator for each subject
        
    #         for sub in subjects:
    #             files_test_sub = [f for f in test_files if sub in os.path.basename(f)]
    #             test_generator = DataGenerator(files_test_sub, window_length)
    #             datasets_test[sub] = create_tf_dataset(test_generator, window_length, batch_equalizer_fn,
    #                                                 hop_length, batch_size=1,
    #                                                 number_mismatch=number_mismatch,
    #                                                 data_types=(tf.float32, tf.float32),
    #                                                 feature_dims=(64, stimulus_dimension))

    #         evaluation = evaluate_model(model, datasets_test)
            # # prediction = predict_model(model, datasets_test)

            # We can save our results in a json encoded file
            # results_path = os.path.join(results_folder, results_filename)
            # with open(results_path, "w") as fp:
            #     json.dump(evaluation, fp)
            # logging.info(f"Results saved at {results_path}")

