# Project - ECG Arrhythmia Classification

## General
This is an obligatory project created for "machine learning" classes taught at the UEP. 
The data were extracted from [Kaggle](https://www.kaggle.com/datasets/sadmansakib7/ecg-arrhythmia-classification-dataset?select=Sudden+Cardiac+Death+Holter+Database.csv), but due to their excessive size they have not been uploaded to GitHub. 
The purpose of the project is to detect arrhythmia based on ECG examination. There are 5 categories (dependent variable):
- N (Normal),
- SVEB (Supraventricular ectopic beat),
- VEB (Ventricular ectopic beat),
- F (Fusion beat),
- Q (Unknown beat).

## Technology
The whole survey is written in Jupyter Notebook with Python language using packages like:
- numpy
- pandas
- seaborn
- matplotlib
- sklearn
- imblearn

## Structure
The project consists of several sections:
- Data exploration

  It involved examining the data for blank and outliers values, visualising them, spliting them into training and test data, creating an additional variable, and resampling data.
  
- Data selection

  User has a possibility to choose which dataset to use in the survey (train, test, sample, resample).
  
- Model

  Four classification models were created. For all of them (exept for dummy model) gridsearch was used to find optimal parameters, which was followed by training data and an assessment of accuracy.
  
    - Dummy model
    - Decision Tree
    - Random Forest
    - K-Nearest Neighbors (KNN)


- Voting Classifier

  Gridsearch has identified the two best results voting soft and hard. It turned out that the Random Forest had the biggest influence on the Voting Classifier.
    

## Citation

- http://ecg.mit.edu/george/publications/mitdb-embs-2001.pdf
- https://www.taylorfrancis.com/chapters/edit/10.1201/9781003028635-11/harnessing-artificial-intelligence-secure-ecg-analytics-edge-cardiac-arrhythmia-classification-sadman-sakib-mostafa-fouda-zubair-md-fadlullah
