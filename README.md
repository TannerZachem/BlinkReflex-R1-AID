
# Blink Reflex R1 AID
This repository contains the code used to generate the Blink Reflex R1 Algorithmic Intraoperative Detection (R1-AID) Model for the manuscript "Interpretable Machine Learning Identification of Blink Reflex responses during Cerebellopontine Angle Surgery".
It provides functionality for 
1. Data Processing (DataProcess.py)
2. Model Training and Testing (R1AID-ML.py)
3. Performance Analysis (ModelAnalysis.py)

## Requirements
To run this code, you will ened the following packages (found in'''requirements.txt'''):
'''
imbalanced_learn==0.12.4
matplotlib==3.10.3
numpy==2.3.1
pandas==2.3.1
scikit_learn==1.7.1
scipy==1.16.0
seaborn==0.13.2
torch==2.3.1
tqdm==4.66.4
'''
Newer package versions will likley work.
Machine-specific installation of [Pytorch is recommended.](https://pytorch.org/get-started/locally/)

## Datasets
At this point we are unable to publically share the raw data and clinical outcomes used to generate this model, please contact '''tanner.zachem@duke.edu''' for more information.  