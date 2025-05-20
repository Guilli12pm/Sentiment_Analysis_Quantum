# Sentiment Analysis with Quantum Circuits QRNNs

This code is used to perform sentiment analysis given a dataset with sentences and a label representing a positive or negative sentiment, using Quantum Machine Learning (QML) Methods.

QML is based on Parametrised Quantum Circuits (PQC) with multiple angle parameters that will be trained to obtain a specific measurement for a quantum circuit. 

The specific QML method used in this project is the Quantum Recurrent Neural Network. 

For the binary classification of the sentiment, there are three methods available:

1) Free Fermionic Quantum circuit: a classically simulable quantum method using matchgates.
2) Fully Quantum Circuit: a generic quantum circuit.
3) Classical Circuit: a generic classical RNN

For each, the number of layers and the hidden size/number of qubits can be changed.

There are two datasets available: Tweets from Twitter and Rotten Tomatoes reviews.

Requirements for this project are in the **requirements.txt** file.

Run with **python3.11**

