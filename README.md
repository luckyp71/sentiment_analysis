# Sentiment Analysis

This project builds two sentiment classification models on a small Chinese dataset consisting of three sentiment labels: positive, neutral, and negative.
Two different approaches are implemented:<br/>
- Classical Machine Learning — Logistic Regression + TF-IDF
- Transformer-based — Fine-tuned BERT (bert-base-multilingual)

## Dataset Overview

Total samples: 2,200.<br/>
Only 18 unique chinese sentences, each repeated 90–151 times.<br/>
Labels: positive, neutral, negative<br/><br/>

Due to the small number of unique texts, the model is expected to memorize rather than generalize.<br/>
The project therefore serves as a testing and reproducibility demonstration.

## Algorithms
## Logistic Regression with TF-IDF
This approach represents a classical machine learning baseline.<br/>
- **TF-IDF (Term Frequency-Inverse Document Frequency)**<br/>
  Convert raw text into numerical vectors that reflect how important each word is relative to the documnt and entire corpus.<br/>
  This helps highlight discriminative words i.e. negative and positive words
- **Logistic Regression**
  A linear classifier well-suited for text classification.<br/>
  It outputs probabilities for each sentiment class and provides stable performance with minimal tuning.

- **Advantages**
  1. Fast to train and easy to interpret<br/>
  2. Requires no GPU<br/>
  3. Performs well on small datasets<br/>
  4. Provides a strong baseline before moving to deep models.
  
- **Trade-offs**
  1. Limited in capturing context and word order (e.g. sarcasm or long dependencies)<br/>
  2. Struggles when the vocabulary or phrasing varies significantly.

## BERT Fine-Tuning
BERT has beecom the standard or text understanding tasks, including sentiment analysis.<br/>
Unlike traditional models that treat words independently, BERT reads the text bidirectionally,
considering the entire contxt before and after each word.<br/>
- Base Model: bert-base-multilingual
  Selected to handle Chinese and potentially multilingual text seamlessly (where the data is available to be used for fine-tuning).<br/>

- **Advantages**  
  1. Captures complex semantic and syntactic relationships
  2. Learns rom larg-scale pretraining, improving accuracy even on small labeled data
  3. Robust to variations in phrasing, word choice, and sentence length

- **Trade-offs**
  1. Computationally expensive (requires GPU for faster training and inference)
  2. Longer training time
  3. Overitting risk when the dataset is too small
  
# Evaluation
## Classical ML Model
Since this is a multi-class classification problem (positive, neutral, negative), 
the models were evaluated using a confusion matrix and key performance metrics:
- Accuracy: overall correctness of predictions
- Precision: how many predicted labels are correct (model reliability)
- Recall (Sensitivity): how many true labels are correctly identified
- F1-score: harmonic mean of precision and recall

All these metrics were computed and visualized in the Jupyter notebook.

**Observation**
All evaluation metrics (accuracy, precision, recall, F1-score) achieved a perfect score of 1.0 across all classes.
This result typically indicate overfitting or data leakage, in this case, the dataset contains only 18 unique sentences repeated many times.<br/>
Therefore, the model is memorizing rather than learning to generalize.


## BERT Model
For the BERT-based model, the evaluation was primarily based on training loss since the dataset size was too small to support a meaningful validation split.
The training loss consistently decreased over time, indicating strong overfitting.<br/><br/>

However, due to the limited dataset, the training results do not reflect the model's true generalization ability.<br/>
Similar to the classical model, BERT most likely memorized the dataset rather than learning robust sentiment representations.

## Model Artefact
- Classical ML Model (Logistic Regression + TF-IDF): ./models/lr_model/
- BERT model: ./models/bert_sentiment_model/

## Model Usage:
Both models can be loaded and used for inference as demonstrated in the accompanying Jupyter Notebook.<br/>
The notebook provides examples for:
- Loading the saved model artefacts
- Performing predictions on new text inputs
- Interpreting sentiment classification outputs


# Environment & Reproducibility
## Python Version
Python 3.10+

## Required Libraries
The following Python libraries were used to train and evaluate the models:<br/>
- pandas and numpy: for data manipulation
- jieba: for Chinese tokenization
- fasttext: for language detection
- scikit-learn, matplotlib, and seaborn: for classical model training and visualization
- joblib: for saving and loading classical ML models
- datasets, transformers, and evaluate: for fine-tuning and evaluating BERT

## Installation
- Craete python virtual environment
- Activate your python virtual environment
- Run this command: pip install -r ./requirements.txt in project root directory

## Reproducing the Training and Inference
- Open the provided Jupyter Notebook (template.ipynb)
- Run all cells to preprocess the data, train models, and evaluate performance
- The trained model artefacts will be automatically saved in the ./models/ directory
- Use the provided code snippets to reload the models and perform inference on new sentences (unsen data).