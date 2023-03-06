# nn_lipo_SMILES

Our goal is to predict LogD (Lipophilicity) given the SMILES of drug molecules by implementing a neural network from scratch. 
We implemented all of the functions needed to initialize, train, evaluate, and make predictions with the network.

Lipophilicity [1] is an important feature of drug molecules that affects both membrane permeability and solubility. This dataset, a part of the lower-N 2D datasets was extracted from MoleculeNet [2] and used as prepared by Yang et al. [3] for regression tasks on lipophilicity (lipo). It is originally curated from ChEMBL database and provides experimental results of octanol/water distribution coefficient (LogD at pH 7.4) of 4200 compounds.

Dataset format The data is available under lipo_dataset, in the files lipo_train.csv, lipo_test.csv that contain molecular SMILES and their corresponding LogD values. The dataset was split randomly using a 85/15 split for training/testing. The two files contain 3570 and 630 instances respectively.
