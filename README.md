This project introduces a system design for the early prediction of Chronic Kidney Disease (CKD) using a deep learning-based classification model. 
The system is designed to process structured clinical data, perform robust preprocessing, and deliver accurate diagnostic predictions with minimal computational overhead. 
The design follows a modular pipeline approach beginning with data ingestion, followed by preprocessing (handling missing values, label encoding, feature scaling), 
model training using a Multi- Layer Perceptron (MLP), and finally, performance evaluation. 
The architecture emphasizes both accuracy and simplicity by utilizing only nine key clinical features such as serum creatinine, hemoglobin, and albumin that are routinely available in basic diagnostic labs. 
The MLP model is implemented using PyTorch and consists of an input layer, two hidden layers with ReLU activation, dropout, and batch normalization, 
and an output layer using sigmoid activation for binary classification. The system is trained using Binary Cross-Entropy loss and optimized with the Adam optimizer. 
The training loop supports mini-batch learning and tracks loss and accuracy per epoch. For testing, the model is evaluated using accuracy, confusion matrix, and other performance metrics.
