# Multi-Layer Perceptron with back-propogation network

## CECS550

## Project2 - MLP

## Team: Incognito

## Team Members:
<ul>
  <li>Aishwarya Bhosale</li>
  <li>Apoorva Uppala</li>
  <li>Dinesh Reddy Kommera</li>
  <li>Keshav Bhojanapally</li>
  </ul>
  
  
  
## Intro:
  A multilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN). An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer
  Description:
  * In this project, we used n-fold cross validation to estimate the skill of a machine learning model on unseen data.
  * Sigmoid function is used for defining the output of the node based on given inputs.
  * 550-01-dataset1.txt consists of dataset which is splited into train, and holdout set. For validation, 550-01-dataset2.txt is used for finding validation accuracy.
  * Updated the initial weights using backpropogation.
  * Number of epochs are determined based on error condition boundary.
  * Displaying confusion matrix for validation dataset.
  

## Contents:
  550-01-p2-Incognito.zip consists of CECS550_Project2.py, CECS550_Project2.ipynb 550-01-dataset1.txt, 550-01-dataset2.txt and Readme.txt
  
## Setup and Installation:

Option 1: Python Cmd
  * Navigate to python downloads page and install python with specified steps.
  * pip install pandas
  * pip install numpy
  * pip install random
  * pip install math
  * pip install matplotlib

Option 2: Jupyter Notebook
  * Install Jupyter Notebook(https://jupyter.org/install)
  * Upload the CECS550_Project2.ipynb
  * Run the file
 
## Sample Invocation:
  python CECS550_Project2.py(cmd as in Option 1)

## Issues:
  
  
## References:
  https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
