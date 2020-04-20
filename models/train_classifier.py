import sys
import re
import pandas as pd
import pickle

from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', con = engine)
    X = df['message'].values
    Y = df[df.columns[4:]].values
    category_names = df.columns[4:]
    
    return X, Y, category_names


def tokenize(text):
	url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
	
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
		('clf', MultiOutputClassifier(KNeighborsClassifier())), #KNN classifier
		])
	
	# parameters for GridSearch
	parameters = {
		'vect__ngram_range': ((1, 1), (1, 2)), 
		#'vect__max_df': (0.5, 0.75, 1.0), 
		#'vect__max_features': (None, 5000), 
		'tfidf__use_idf': (True, False),
		#'clf__estimator__n_neighbors': [5, 10], 
		'clf__estimator__p': [1, 2]
		}
	
	model = GridSearchCV(pipeline, param_grid = parameters, verbose = 10)
	return model

def evaluate_model(model, X_test, Y_test, category_names):
	Y_pred = model.predict(X_test)
	labels = np.unique(y_pred)
	confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
	accuracy = (y_pred == y_test).mean
	print("Labels:", labels)
	print("Confusion Matrix:\n", confusion_mat)
	print("Accuracy:", accuracy)
	
	for cat in category_names:
		print('f1 score for {}: {}'.format(cat, f1_score(Y_test[cat], Y_pred[cat]))


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
