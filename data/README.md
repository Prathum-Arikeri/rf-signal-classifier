# Dataset Overview

This folder contains raw RF signal datasets used for training and testing the RF signal classifier.

## Contents

- **RadioML2016.10a**: Public dataset of I/Q samples labeled by modulation type.  
  Source: https://github.com/radioml/datasets  
  Format: `.npy` files with I/Q samples, accompanying CSV metadata with labels.

- **Synthetic Signals**: Generated using custom Python scripts simulating AM, FM, PSK, QAM signals.  
  Format: `.npy` arrays saved with sampling rate and modulation metadata.

## Usage

- The raw data is unprocessed and should be preprocessed using `scripts/preprocess.py` before training.
- Data files are named with their modulation type and SNR level for easy filtering.
- Due to file sizes, only subsets are included in this repo. Full datasets can be downloaded from [source links].

## Notes

- Ensure data privacy and compliance when using third-party datasets.
- Future additions may include more radar or spectrum datasets.
