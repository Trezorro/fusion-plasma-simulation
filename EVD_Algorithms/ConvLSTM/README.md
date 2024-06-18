# CNN-LSTM

Implementation of the CNN-LSTM model for classification of plasma confinement states.

NF publication 2020 by F. Matos: 
https://iopscience.iop.org/article/10.1088/1741-4326/ab6c7a

The model architecture presented in the following code is not the same as the one shown in the publication.
It was highly pruned, reducing #params by a factor of 36, which led to an improvement of +1%,+8%,+3% for L,D and H modes respectively.The current architecture is the one used in the real-time PCS of TCV.


## Installation

<b># Installation of Miniconda from scratch</b>
- Get and install Miniconda:
    1. `cd your_project/` (Miniconda packages might require a significant space ~Gbs)
    1. `wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh`
    2. `bash Miniconda3-latest-Linux-x86_64.sh`
    3. `export PATH="/home/user/your_project/miniconda3/bin:$PATH"` (or where you have decided to install miniconda3)

<b># Create an environment (GPU)</b>
- An environment file is provided for version compatibility.\
`conda create --name my_gpu_env --file environment_gpu.txt`\
`pip install tensorflow-gpu==2.3`\
Please use this tf version since older ones, as tf 2.2, had some time consuming warnings when doing model predictions\
as explained here https://github.com/tensorflow/tensorflow/issues/41347

<b># Create an environment (CPU)</b>
- Once you have installed conda locally, you can setup a centralized ML environment in LACs:\
`conda activate ml_env`

## Preparation of Experiments
### Obtain dataset

<b>Dataset</b>
- The most up-to-date validated files are in Lac8_D partition from lac8:\
`/Lac8_D/DISTOOL/JET/Event_Detection/`
- For TCV there are two validated versions `dense` and `loose`. The first one labels very short LD-DL transitions while the second one
considers just one label (L or D) which results to be more practical and yields a less noisy model.\
`/Lac8_D/DISTOOL/JET/Event_Detection/TCV_Validation/LHD/baseline_validation_loose/`
- Filenames must have the following format: "Machine_ShotNumber_Labeler.csv", e.g: "TCV_26386_apau_and_marceca_labeled.csv"
- To start training your model just copy the validated shots in your local folder:\
`cp /Lac8_D/DISTOOL/JET/Event_Detection/TCV_Validation/LHD/baseline_validation_loose/TCV_*_LABELER_labeled.csv ./labeled_data/TCV/LABELER/`\
Currently the LABELER name is under `marceca`, but the data was consistently validated by both Alessandro Pau and Gino Marceca

### Visualize data inputs from generator
`python lstm_data_generator.py`

### Run an experiment
`(python lstm_train.py #experiment #machine #labeler)`\
`python lstm_train.py 2 TCV marceca`

### Compute predictions
`(python lstm_predict.py #experiment #epoch #test_machine)`\
`python lstm_predict.py 2 400 JET`

### Evaluate Kappa scores
`(python lstm_scores.py #experiment #epoch #test_machine)`\
`python lstm_scores.py 2 400 JET`

### Predict for a given shot and save in DIS_tool format
`python algorithms/ConvLSTM/evaluate_model_from_detected_signals.py #experiment #epoch #shot #target_machine #normalization_method #git_test`\
python algorithms/ConvLSTM/evaluate_model_from_detected_signals.py baseline_16042021_exp9 400 70454 TCV minmax True
