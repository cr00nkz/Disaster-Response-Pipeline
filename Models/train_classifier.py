# import libraries
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
import pickle
nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(db="sqlite:///../Pipeline/ETL_Disaster.db", table="DISASTER_DATA"):
    """Loads the previously saved data from database and table

    INPUT: db - The sqlite database file 
                (default: sqlite:///../Pipeline/ETL_Disaster.db)
           table - The tablename containing the data (default: DISASTER_DATA)

    RETURNS: X - Dataframe containing the messages for training/testing
             Y - Dataframe containing the 36 predictions for X
    """
    # load data from database
    engine = create_engine(db)
    df = pd.read_sql_table(table, engine)

    #prepare X and y dataframes 
    X = df["message"]
    y = df.drop(columns=["id", "message", "original", "genre"])

    return X, y


def tokenize(text):
    """The tokenizer used for the CountVectorizer, 
        which is called in the pipeline.
        Performs the following actions on the input text:
        * convert to lowercase
        * remove punctuation
        * tokenize
        * remove stopwords
        * lemmatize
        * stem

    INPUT: text - The text to tokenize

    RETURNS: A processed tokenized array
    """
    #remove punctuation, lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #tokenize
    words = nltk.tokenize.word_tokenize(text)
    
    #remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Lemmatize our words
    words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
    
    # Stem our words
    words = [PorterStemmer().stem(w).strip() for w in words]
    return words


def build_model(create_simple=False):
    """Prepare a pipeline for fitting and prediction

    INPUT: create_simple: simple model for fast training (default: False)

    RETURNS: A GridSearchCV model, which needs to be fitted.
    """

    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {"vect__ngram_range": [(1, 1), (1, 2)],
                "vect__max_df": (0.5, 1.0),
                "clf__estimator__max_features": ['auto', 'sqrt'],
                #"clf__estimator__bootstrap": [True, False],
                #"clf__estimator__max_depth": [10, 30, 50, 70, 90, None],
                #"clf__estimator__min_samples_split": [2, 5, 10],
                #"clf__estimator__n_estimators": [200, 600, 1000, 1400, 1800]}
                "clf__estimator__min_samples_leaf": [1, 2]}
            
    if (create_simple):
        parameters = {"vect__ngram_range": [(1, 1), (1, 2)]}

    model = GridSearchCV(pipeline, parameters, verbose=2, cv=2)

    """These values were returned for the complex model 

    (cv.best_estimator_.steps)
    [('vect', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=0.5, max_features=None, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words=None,
        strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
        tokenizer=<function tokenize at 0x7fdf6a7e6268>, vocabulary=None)), ('tfidf', TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)), ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='sqrt', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False),
            n_jobs=1))]"""

    """These values were returned for the simple model 

    (cv.best_estimator_.steps)
    [('vect', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
            tokenizer=<function tokenize at 0x7fdf6a7e6268>, vocabulary=None)),
    ('tfidf',
    TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)),
    ('clf',
    MultiOutputClassifier(estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False),
                n_jobs=1))]"""


    return model


def train_predict_eval(model, X, y):
    """Trains, predicts and evaluates the model.

    INPUT: model - The model (pipeline) to train
           X - The data to train and predict on the model
           y - The results to train and evaluate the model

    RETURNS: The trained model
    """
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # fit model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    # output model test results
    for i in range(y_test.shape[1]):
        rep = classification_report(y_test.iloc[:, i], y_pred[:, i])
        rep = rep.split("avg / total")[1].split("     ")
        print("Column \"{}\": Precision: {}, Recall: {}, F1-Score: {}"
                .format(y_test.columns[i], rep[1].strip(), 
                    rep[2].strip(), rep[3].strip()))

    return model


def export_model(model, filename="model.pck"):
    """Exports the model to a pickle file

    INPUT: model - The model to be pickled/saved
           filename - The name of the file to be saved (default: model.pck)
    """

    # Export model as a pickle file
    dumpfile = open(filename, "wb")
    pickle.dump(model, dumpfile)
    dumpfile.close()

if __name__ == "__main__":
    X, y = load_data() # Load data from DB
    print("Loaded data")
    model = build_model(True) # Build pipeline model
    print("Built model")
    model = train_predict_eval(model, X, y) # Train model
    print("Trained model")
    export_model(model) # Save model to HDD
    print("Exported model to HDD")

