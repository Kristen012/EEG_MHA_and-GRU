"""Experiments for different models."""
import glob
import json
import logging
import os
import tensorflow as tf

# (un)comment the desired model(s)
#from experiment_models import dilation_model                 # Baseline model
#from experiment_models import eeg_mha_dc_speech_dc_model     # MHA+DC for EEG and DC for speech stimulus
from experiment_models import eeg_mha_dc_speech_gru_dc_model, TransformerBlock # MHA+DC for EEG and GRU+DC for speech stimulus
from EEGNet import EEGNet
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


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Select GPU number 1
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    # Parameters
    # Length of the decision window
    window_length_s = 5

    window_length = 5 * 64  # 3 seconds
    # Hop length between two consecutive decision windows
    hop_length = 64
    # Number of samples (space) between end of matched speech and beginning of mismatched speech
    spacing = 64
    epochs = 100 #100
    patience = 6  #Stops training after 6 epochs without improvements
    lr_decay_patience = 2 #Decrease learning rate after x epochs without improvements
    batch_size = 32
    only_evaluate = False
    number_mismatch = 4 # or 4

    training_log_filename = "training_log_{}_{}.csv".format(number_mismatch, window_length_s)
    # results_filename = 'eval.json'


    # Get the path to the config file
    experiments_folder = os.path.dirname(__file__)

    # Provide the path of the dataset
    # which is split already to train, val, test (split_data folder)
    data_folder = 'split_data'

    # The name of the current training run (a respective folder to save the model checkpoint, the training logs, and evaluation results will be created)
    train_run_name = 'TRAIN_RUN_NAME'

    # stimulus feature which will be used for training the model. Can be either 'envelope' (dimension 1) or 'mel' (dimension 28)
    # stimulus_features = ["envelope"]
    # stimulus_dimension = 1
  
    # uncomment if you want to train with the mel spectrogram stimulus representation
    stimulus_features = ["mel"]
    stimulus_dimension = 28

    features = ["eeg"] + stimulus_features

    # Create a directory to store (intermediate) results
    results_folder = os.path.join(experiments_folder, "results_dilated_convolutional_model_three_head_{}_MM_{}_s_{}".format(number_mismatch, window_length_s, stimulus_features[0]))
    os.makedirs(results_folder, exist_ok=True)

    # create model - (un)comment the desired model(s)
    #model = dilation_model(time_window=window_length, eeg_input_dimension=64, env_input_dimension=stimulus_dimension)                 # Baseline model
    #model = eeg_mha_dc_speech_dc_model(time_window=window_length, eeg_input_dimension=64, env_input_dimension=stimulus_dimension)     # MHA+DC for EEG and DC for speech stimulus
    model = eeg_mha_dc_speech_gru_dc_model(time_window=window_length, eeg_input_dimension=64, env_input_dimension=stimulus_dimension, num_mismatched_segments = number_mismatch) # MHA+DC for EEG and GRU+DC for speech stimulus
    # model = EEGNet(time_window=window_length, eeg_input_dimension=64, env_input_dimension=stimulus_dimension, num_mismatched_segments = number_mismatch)
    # tf.keras.utils.register_keras_serializable(TransformerBlock)
    
    model_path = os.path.join(results_folder, "model_{}_MM_{}_s_{}.h5".format(number_mismatch, window_length_s, stimulus_features[0]))

    if only_evaluate:
        model = tf.keras.models.load_model(model_path, custom_objects={"TransformerBlock": TransformerBlock})

    else:
        train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        # print(data_folder)
        # print(train_files)
        
        # Create list of numpy array files
        train_generator = DataGenerator(train_files, window_length)
        import pdb
        dataset_train = create_tf_dataset(train_generator, window_length, batch_equalizer_fn,
                                          hop_length, batch_size,
                                          number_mismatch=number_mismatch,
                                          data_types=(tf.float32, tf.float32),
                                          feature_dims=(64, stimulus_dimension))

        # Create the generator for the validation set
        val_files = [x for x in glob.glob(os.path.join(data_folder, "val_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        val_generator = DataGenerator(val_files, window_length)
        dataset_val = create_tf_dataset(val_generator,  window_length, batch_equalizer_fn,
                                          hop_length, batch_size,
                                          number_mismatch=number_mismatch,
                                          data_types=(tf.float32, tf.float32),
                                          feature_dims=(64, stimulus_dimension))


        # Train the model
        model.fit(
            dataset_train,
            epochs=epochs,
            validation_data=dataset_val,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
                tf.keras.callbacks.CSVLogger(os.path.join(results_folder, training_log_filename)),
                tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
            ],
        )
    
    test_window_lengths = [3,5]
    number_mismatch_test = [2,3,4, 8]
    for number_mismatch in number_mismatch_test:
        for window_length_s in test_window_lengths:
            window_length = window_length_s * 64
            results_filename = 'eval_{}_{}_s.json'.format(number_mismatch, window_length_s)

            model = eeg_mha_dc_speech_gru_dc_model(time_window=window_length, eeg_input_dimension=64,
                                   env_input_dimension=stimulus_dimension, num_mismatched_segments=number_mismatch)

            model.load_weights(model_path)
            # Evaluate the model on test set
            # Create a dataset generator for each test subject
            test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if
                          os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
            # Get all different subjects from the test set
            subjects = list(set([os.path.basename(x).split("_-_")[1] for x in test_files]))
            datasets_test = {}
            # Create a generator for each subject
            for sub in subjects:
                files_test_sub = [f for f in test_files if sub in os.path.basename(f)]
                test_generator = DataGenerator(files_test_sub, window_length)
                datasets_test[sub] = create_tf_dataset(test_generator, window_length, batch_equalizer_fn,
                                                       hop_length, batch_size=1,
                                                       number_mismatch=number_mismatch,
                                                       data_types=(tf.float32, tf.float32),
                                                       feature_dims=(64, stimulus_dimension))

            evaluation = evaluate_model(model, datasets_test)

            # We can save our results in a json encoded file
            results_path = os.path.join(results_folder, results_filename)
            with open(results_path, "w") as fp:
                json.dump(evaluation, fp)
            logging.info(f"Results saved at {results_path}")