import pandas as pd
import os
import dotenv
dotenv.load_dotenv()

train = pd.read_csv(os.environ['TRAIN'])
print("Training Set:"% train.columns, train.shape, len(train))
test = pd.read_csv(os.environ['TEST'])

import re

#Step 1: Data Cleaning
def clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    #Use lambda function to remove all junk text
    df[text_field] = df[text_field].apply(lambda element: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", element))
    return df

test_clean = clean_text(test, "tweet")
train_clean = clean_text(train, "tweet")

#Resample to balance the dataset based on the degree of hate speech
from sklearn.utils import resample
train_majority = train_clean[train_clean.label==0]
train_minority = train_clean[train_clean.label==1]
train_minority_upsampled = resample(train_minority, 
                                 replace=True,    
                                 n_samples=len(train_majority),   
                                 random_state=123)
train_upsampled = pd.concat([train_minority_upsampled, train_majority])
train_upsampled['label'].value_counts()

#Step 2: Data Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
pipeline_sgd = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf',  TfidfTransformer()),
    ('nb', SGDClassifier()),])

#Step 3: Feature Engineering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_upsampled['tweet'],
                                                    train_upsampled['label'],
                                                    random_state=0)
#Use TF-IDF to vectorize the text
vect = TfidfVectorizer().fit(X_train)
X_train_vectorized = vect.transform(X_train)

#Train the model
model = pipeline_sgd.fit(X_train, y_train)

#Predict the results
predictions = model.predict(X_test)

#Step 4: Model Evaluation
from sklearn.metrics import *
print("Accuracy:", accuracy_score(y_test, predictions))
#print(f1_score(y_test, predictions, average="macro"))
print(classification_report(y_test, predictions))



