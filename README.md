# Emotion Recognition from EEG Signals


## Purpose
Project related to the Natural Interaction and Models of Affective and Behavioral Computing courses for the Computer Science Master at Università degli Studi di Milano.

This repository is intended to keep trace of the project development, storing both the code and the project report.

## Overview
In this project I dealt with emotion recognition from EEG signals using a SVM classifier based on libsvm library, relying on the method described in the article https://www.hindawi.com/journals/bmri/2017/8317357/. 

This method consists in making classification and prediction using features extracted from decomposed EEG signals.

First of all, EEG signals taken from the DEAP dataset are decomposed into IMFs (Intrinsic Mode Functions) using the EMD (Empirical Mode Decomposition), then the first difference of time series, the first difference of phase and the normalized energy are extracted as features from the first IMF, the most informative one. Two disjoint sets are formed using the extracted features: a training set and a test set. Classification was made on the training set and the respective label set; then, prediction was made on the test set, to test the trained model on a different set without giving it the labels as input. Predicted values was confronted with the labels to check if the model prediction was right or wrong.
Using cross-validation, it was possible to extimate the model accuracy for a single participant in the DEAP dataset. Finally the method accuracy was computed as the mean of the accuracies computed on the single participants.
