import sys
import re
import pandas as pd
import pickle

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{}')
    df = pd.read_sql_table('messages', con = engine)
    X = df['message'].values
    Y = df[df.columns[4:]].values
    category_names = df.columns[4:]
    
    return X, Y, category_names


def tokenize(text):
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok, pos = 'v').lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # build a pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)), #transformer
        ('tfidf', TfidfTransformer()), #transformer
        ('clf', MultiOutputClassifier()), #classifier
        ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    # evaluate all steps on test set
    predicted = model.predict(X_test)
    
    parameters = {
        'scaler__with_mean': [True, False]
        'clf__kernel': ['linear', 'rbf'],
        'clf__C':[1, 10]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    pass

def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)


def save_model(model, model_filepath):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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