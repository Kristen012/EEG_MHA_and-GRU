Auditory EEG Challenge - ICASSP 2024
================================
Our model use Multi-Head Self Attention and Gated Recurrent Unit for baseline model improving (just a little bit...)
## General Setup
### 1. Clone this repository and install the [requirements.txt](requirements.txt)
```bash
# Clone this repository
git clone https://github.com/Kristen012/EEG_MHA_and-GRU.git

# Go to the root folder
cd EEG_MHA_and-GRU

# Optional: install a virtual environment
python3 -m venv venv # Optional
source venv/bin/activate # Optional

# Install requirements.txt
python3 -m install requirements.txt
```

### 2. [Download the data](https://homes.esat.kuleuven.be/~lbollens/)

You will need a password, which you will receive when you [register](https://exporl.github.io/auditory-eeg-challenge-2024/registration/).
The folder contains multiple folders (and `zip` files containing the same data as their corresponding folders). For bulk downloading, we recommend using the `zip` files, 

   1. `split_data(.zip)` contains already preprocessed, split and normalized data; ready for model training/evaluation. 
If you want to get started quickly, you can opt to only download this folder/zipfile.

   2. `preprocessed_eeg(.zip)` and `preprocessed_stimuli(.zip)` contain preprocessed EEG and stimuli files (envelope and mel features) respectively.
At this stage data is not yet split into different sets and normalized. To go from this to the data in `split_data`, you will have to run the `split_and_normalize.py` script ([preprocessing_code/split_and_normalize.py](./preprocessing_code/split_and_normalize.py) )

   3. `sub_*(.zip)` and `stimuli(.zip)` contain the raw EEG and stimuli files. 
If you want to recreate the preprocessing steps, you will need to download these files and then run `sparrKULee.py` [(preprocessing_code/sparrKULee.py)](./preprocessing_code/sparrKULee.py) to preprocess the EEG and stimuli and then run the `split_and_normalize.py` script to split and normalize the data.
It is possible to adapt the preprocessing steps in `sparrKULee.py` to your own needs, by adding/removing preprocessing steps. For more detailed information on the pipeline, see the [brain_pipe documentation](https://exporl.github.io/brain_pipe/).


Note that it is possible to use the same preprocessed (and split) dataset for both task 1 and task 2, but it is not required.

## Model performance
- validation acc
<table>
  <tr>
    <td>baseline</td>
    <td>MHA+GRU</td>
    <td>MHA+GRU(with additional batch normalization)</td>
    <td>MHA+GRU(with addtional mel preprocess)</td>
  </tr>
  <tr>
    <td>58%</td>
    <td>61.9%</td>
    <td>61.6%</td>
    <td>61.ㄞ%</td>
  </tr>
</table>
## Model output examples
After the training process complete, you will get folowing files under the `results_dilated_convolutional_model_{number_mismatch}_MM_{decision_window_length}_s_{stimulus_features}` folder.

1. eval_{number_mismatch}_{decision_window_length}_s.json
2. training_log_{number_mismatch}_{decision_window_length}
3. model_{number_mismatch}_MM_{decision_window_length}_s_{decision_window_length}.h5

- Refer to [results_dilated_convolutional_model_4_MM_5_s_dim28_mel](results_dilated_convolutional_model_4_MM_5_s_dim28_mel)
