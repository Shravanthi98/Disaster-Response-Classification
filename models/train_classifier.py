# Import all the necessary libraries.
import sys
import re
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import fbeta_score, make_scorer

def load_data(database_filepath):
    """
    Function to load the given data from SQL Database
    Arguments:
    database_filepath: Path to the location of the database. E.g. Disaster.db
    
    Returns input, X and ground truth labels, Y.
    """
    # Create an engine to load the data
    engine = create_engine('sqlite:///'+ database_filepath)
    # Read the data from the SQL table
    df = pd.read_sql_table('Disaster', engine)
    # Extract the input and labels for training
    X = df['message']
    Y = df.iloc[:, 3:]
    return X, Y

def tokenize(text):
    """
    Function used to clean and process text data.
    Arguments:
    text: Input text that is to be cleaned and tokenized.
    
    Returns a list of clean tokens.
    """
    # Regex syntax to detect URLs
    re_url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Pull all the URLs from the given text 
    detected_urls = re.findall(re_url, text)
    
    # Replace each url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, "urlplaceholder")

    # Convert text to tokens
    tokens = word_tokenize(text)
    # Instantiate the lemmatizer class
    lemmatizer = WordNetLemmatizer()

    # List to store all the processed tokens
    clean_tokens = []
    # For each token, lemmatize, normalize, and strip out the extra white spaces.
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Function to define and build the model by extracting appropriate features and finding optimal parameters through Grid Search.
    
    Returns the pipeline/estimator.
    """
    # Define the pipeline with feature extractor, classifier.
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # List of parameters to use in Grid Search.
##    parameters = {
##    'vect__max_df': (0.5, 0.75, 1.0),
##    'clf__estimator__n_estimators': [10, 20],
##    'clf__estimator__min_samples_split': [2, 3, 4],
##    }
##
##    # Using Grid Search to find the optimal set of parameters based on F-score.
##    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=make_scorer(calculate_fscore, beta=1))

    return pipeline


def calculate_fscore(y_true, y_pred, beta):
    """
    Function used to calculate the average F-score of all the labels for Grid Search.
    Arguments:
    y_true: True labels
    y_pred: Predicted labels
    beta: weight of recall in the combined score. E.g. 0.5, 1, 2 are the commonly used values.
    
    Returns the average F-score.
    """
    # Convert the DataFrame into an array.
    y_true = y_true.values
    fscore_values = []
    # For each column, calculate the F-score and store it in a list.
    for column in range(y_pred.shape[1]):
        f_score = fbeta_score(y_true[:, column], y_pred[:, column], beta=beta, average='weighted')
        fscore_values.append(f_score)

    # Calculate the average f-score value for all the columns.
    average_fscore = sum(fscore_values)/len(fscore_values)
    return average_fscore


def evaluate_model(model, X_test, Y_test):
    """
    Function used to evaluate the model against the test data.
    Arguments:
    model: Trained model/pipeline
    X_test: Test Dataset
    Y_test: True labels

    Prints the evaluation metrics for each label along with the overall accuracy and F-score.
    """
    # Generate the predictions for the test data
    y_preds = model.predict(X_test)

    # Calculate average accuracy and fbeta-score.
    avg_accuracy = (y_preds == Y_test).mean().mean()
    avg_fscore = calculate_fscore(Y_test, y_preds, beta=1)

    print("Overall Accuracy: {0:.2f}%".format(avg_accuracy*100))
    print("Average F-score: {0:.2f}%".format(avg_fscore*100))

    # For each label, print the classification report with the calculated metrics.
    for column in range(Y_test.shape[1]):
        print("For category: ", Y_test.columns[column])
        print(classification_report(Y_test.iloc[:, column], y_preds[:, column]))
    

def save_model(model, model_filepath):
    """
    Function to save the final model using pickle.
    Arguments:
    model: Trained model.
    model_filepath: Path to store the trained model (.pkl file)
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
