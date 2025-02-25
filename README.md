# CNN Text Classification

## Overview
This project builds a **Convolutional Neural Network (CNN)** model for classifying text into different emotional categories. The workflow includes data preprocessing, tokenization, model training, evaluation, and prediction.

## Installation
To run the project, ensure you have the following dependencies installed:
```bash
pip install numpy pandas nltk tensorflow scikit-learn
```

## Data Preprocessing
- Convert text to lowercase.
- Remove special characters, numbers, and extra spaces.
- Tokenize text into words.
- Remove stopwords.
- Perform lemmatization.
- Remove duplicate sentences.

## Model Architecture
The CNN model consists of:
- **Embedding Layer**: Converts words into vector representations.
- **1D Convolutional Layer**: Extracts text patterns.
- **Global Max Pooling**: Reduces feature dimensions.
- **Dense Layers**: Fully connected layers for classification.
- **Dropout Layer**: Prevents overfitting.
- **Output Layer**: Uses softmax activation for multi-class classification.

## Training & Evaluation
- **Loss Function**: `sparse_categorical_crossentropy`
- **Optimizer**: `Adam`
- **Metric**: `Accuracy`
- **Epochs**: 5
- **Batch Size**: 32

## Prediction & Submission
The trained model predicts emotions from new text data and saves the results in `submission_cnn.txt`.

## Future Improvements
- Utilize **pre-trained embeddings** (e.g., GloVe, Word2Vec).
- Combine **CNN with LSTM/GRU** for better context understanding.
- Apply **data augmentation** to enhance dataset size and diversity.

## Usage
Run the script to preprocess data, train the model, and generate predictions:
```bash
python train.py
```

## License
This project is open-source and free to use under the MIT License.

