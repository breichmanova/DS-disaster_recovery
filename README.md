# Disaster Response Pipeline
Project n.2 for Udacity Data Scientist Nanodegree.

### Description
This application loads data from a CSV file, saves it in a SQLite database, transforms the data and finds the best model among the specified parameter combinations by GridSearch. Then this model is used in a Flask web-app, which consists of two parts - a dashboard with graphs and a message classifier.

### Model pipeline
In the ML pipeline I am using transformers: 
* CountVectorizer with self-defined tokenize function 
* TfidfTransformer

and a classifier:
* MultiOutputClassifier with DecisionTreeClassifier
* GridSearchCV

### How to run
* Process the data:
``` python 
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

* Train the model:
``` python 
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

* Run the Flask web-app:
``` python 
python app/run.py
```