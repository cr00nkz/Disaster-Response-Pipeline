import json
import plotly
import pandas as pd
import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet', 'stopwords'])

app = Flask(__name__)

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


# load data
engine = create_engine('sqlite:///../data/ETL_Disaster.db')
df = pd.read_sql_table('DISASTER_DATA', engine)

# load model
model = joblib.load("../models/model.pck")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    print("starting")
    app.run(host='0.0.0.0', port=3001, debug=True)
    print("started")


if __name__ == '__main__':
    main()