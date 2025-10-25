# QKAN-based Intrusion Detection System (IDS)

This project implements an Intrusion Detection System (IDS) using a Quantum-inspired Kolmogorov-Arnold Network (QKAN) autoencoder. The system is designed to detect network intrusions by learning the normal patterns of network traffic and identifying anomalies.

## Features

-   **Two Autoencoder Models:** Includes implementations of both a QKAN autoencoder and a standard MLP autoencoder for comparison.
-   **Support for Multiple Datasets:** The system can be trained and evaluated on the CSE-CIC-IDS2018 and UNSW-NB15 datasets.
-   **Modular Design:** The code is organized into separate modules for data preprocessing, model training, and evaluation, making it easy to extend and modify.
-   **GPU Acceleration:** The training process can be accelerated using a CUDA-enabled GPU.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ailabteam/qkan-ids.git
    cd qkan-ids
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preprocessing

Before training the model, you need to preprocess the data. The following scripts are provided for this purpose:

-   **CSE-CIC-IDS2018:** `preprocess_data.py`
-   **UNSW-NB15:** `preprocess_unsw.py`

To run the preprocessing script for the CSE-CIC-IDS2018 dataset, use the following command:

```bash
python preprocess_data.py
```

### 2. Model Training

The `train.py` script is used to train the autoencoder models. You can specify the model type (`qkan` or `mlp`) and the dataset (`ids2018` or `unsw`) as command-line arguments.

For example, to train the QKAN model on the CSE-CIC-IDS2018 dataset, use the following command:

```bash
python train.py qkan ids2018
```

### 3. Model Evaluation

The `evaluate.py` script is used to evaluate the performance of the trained models. You can specify the model type and dataset to evaluate.

```bash
python evaluate.py qkan ids2018
```

## Models

### QKAN Autoencoder

The QKAN autoencoder is a novel architecture that uses a quantum-inspired Kolmogorov-Arnold Network to learn a compressed representation of the input data. This model is designed to be more efficient and expressive than traditional autoencoders.

### MLP Autoencoder

The MLP autoencoder is a standard multi-layer perceptron autoencoder that is used as a baseline for comparison with the QKAN model.

## Datasets

### CSE-CIC-IDS2018

The CSE-CIC-IDS2018 dataset is a large-scale, realistic dataset for intrusion detection that includes a wide range of modern network attacks.

### UNSW-NB15

The UNSW-NB15 dataset is another widely used dataset for intrusion detection that contains a variety of attack scenarios.

## Results

The performance of the models is evaluated based on their ability to reconstruct normal network traffic while producing high reconstruction errors for anomalous traffic. The evaluation script will output the receiver operating characteristic (ROC) curve and the area under the curve (AUC) for each model.
