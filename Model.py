# Importing Libraries:
import pandas as pd
import numpy as np
import pickle

# for displaying all feature from dataset:
pd.pandas.set_option('display.max_columns', None)

# Reading Dataset:
dataset = pd.read_csv("Stroke_data.csv")

# Dropping unneccsary feature :
dataset = dataset.drop('id', axis=1)

# Filling NaN Values in BMI feature using mean:
dataset['bmi'] = dataset['bmi'].fillna(dataset['bmi'].median())

# Dropping Other gender
Other_gender = dataset[dataset['gender'] == 'Other'].index[0]
dataset = dataset.drop(Other_gender, axis=0)

# Rename some names in worktype feature for simplacity nothing else:
dataset.replace({'Self-employed' : 'Self_employed'}, inplace=True)
# Rename some names in smokers feature for simplacity nothing else:
dataset.replace({'never smoked':'never_smoked', 'formerly smoked':'formerly_smoked'}, inplace=True)

# Dependent & Independent Feature:
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Label Encoding:
X['ever_married'] = np.where(X['ever_married']=='Yes',1,0)   ## If married replace with by 1 otherwise 0.
X['Residence_type'] = np.where(X['Residence_type']=='Rural',1,0)    ## If residence type is Rural replace it by 1 otherwise 0.

# One Hot Encoding:
X = pd.get_dummies(X, drop_first=True)

# Rearranging the columns for better understanding
X = X[['gender_Male','age', 'hypertension', 'heart_disease', 'ever_married',
       'Residence_type', 'avg_glucose_level', 'bmi',
       'work_type_Never_worked', 'work_type_Private','work_type_Self_employed', 'work_type_children',
       'smoking_status_formerly_smoked', 'smoking_status_never_smoked','smoking_status_smokes']]


# Over Sampling:
from imblearn.over_sampling import RandomOverSampler
oversampler = RandomOverSampler(0.4)
x_oversampler, y_oversampler = oversampler.fit_resample(X,y)

# Train Test Split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_oversampler,y_oversampler, test_size=0.2, random_state=0)

# RandomForestClassifier:
from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier()
RandomForest = RandomForest.fit(X_train,y_train)

# Creating a pickle file for the classifier
filename = 'Stroke.pkl'
pickle.dump(RandomForest, open(filename, 'wb'))