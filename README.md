# Climate Change Sentiment Analysis on Twitter

This repository contains the code and resources for our research on **analyzing public sentiment toward climate change on Twitter** using both **traditional machine learning models** and **transformer-based deep learning models**. The research focuses on comparing the performance of these approaches in understanding public opinion, with implications for AI research and environmental communication.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Preprocessing](#preprocessing)
- [Experiments](#experiments)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Overview
The project investigates how effectively different NLP models can classify climate change-related tweets into **positive, neutral, or negative sentiment**. Traditional machine learning models (Logistic Regression, Ridge Classifier, Multinomial Naive Bayes, Random Forest) are compared with transformer-based models (RoBERTa-large-MNLI, BERTweet-base, ClimateBERT-distil). The goal is to identify models that provide both high accuracy and contextual understanding of Twitter text, which often includes emojis, hashtags, and informal language.

## Dataset
The dataset used in this research is the [Twitter Climate Change Sentiment Dataset](https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment-dataset/code) from Kaggle.  
- **Size:** 43,943 tweets  
- **Labels:** -1 (Negative), 0 (Neutral), 1 (Positive), 2 (News)  
- **Language:** Multilingual (filtered to English for this study)  
- **Preprocessing:** Duplicate removal, symbol normalization, URL/mention removal, stopwords removal, lemmatization, and filtering out the "news" category.  

## Models
### Traditional Machine Learning
- Logistic Regression  
- Ridge Classifier  
- Multinomial Naive Bayes  
- Random Forest  

Each model is trained with three feature extraction methods:
- **CountVectorizer** (with n-grams)  
- **TF-IDF**  
- **Word2Vec** (trained on the dataset)

### Transformer-based Deep Learning
- RoBERTa-large-MNLI (fine-tuned)  
- BERTweet-base (fine-tuned)  
- ClimateBERT-distil (fine-tuned)

Training uses tokenization specific to each model, and evaluation metrics include Accuracy, Precision, Recall, and Macro-F1 score.

## Preprocessing
Preprocessing pipelines include:
1. Traditional ML pipeline: lowercase, stopwords removal, stemming/lemmatization, oversampling (random or SMOTE-Tomek), and feature extraction.  
2. Kaggle-based pipeline: CountVectorizer with optimized n-grams.  
3. Deep learning pipeline: tokenization, lowercase, stopwords removal, lemmatization, and SMOTE-Tomek for certain experiments.

## Experiments
- Comparative evaluation of traditional ML vs transformer models.  
- Fine-tuning of BERT-based models using cleaned dataset.  
- Evaluation on 80:20 train-test split using classification metrics.

## Results
- **Best ML baseline:** Ridge + CountVectorizer (unigram-bigram), F1-weighted: 0.77 
- **Best Transformer:** BERTweet, superior Macro-F1 and contextual understanding.  
- **Observations:** Transformer models handle informal Twitter text better (emojis, sarcasm, ALLCAPS), while ML baselines remain competitive on well-preprocessed text.

## Future Work
- Expand dataset via X (Twitter) scraping and manual labeling.  
- Experiment with LLMs using PEFT or LoRA for efficient fine-tuning.  
- Apply explainability methods (SHAP, LIME) to interpret model predictions.  
- Explore multilingual datasets and advanced augmentation for better generalization.

## License
This repository is licensed under the MIT License. See the LICENSE file for details.
