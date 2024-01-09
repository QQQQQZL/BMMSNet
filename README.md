# README

This is the code for BMMSNet. This codebase contains some code from the organizer [exporl/auditory-eeg-challenge-2024-code (github.com)](https://github.com/exporl/auditory-eeg-challenge-2024-code).



# Prerequisites

To run the code, you need to do the following:

0. Python == 3.7

1. Install the the requirements.
2. Download the official dataset following [exporl/auditory-eeg-challenge-2024-code (github.com)](https://github.com/exporl/auditory-eeg-challenge-2024-code). In this work we use the "split_data". 



# Run the code

1. If you want to use the concatenation of mel-spectogram and speech envelope as the input stimulus, first run the `preprocessing_code/get_melenv.py`. You should change the dataset path according to you enviroment first.
2. Run `run.sh` to train valid the model. (By default, all GPUs are used.)



