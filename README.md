# Disaster-Response-Pipeline
A repository consisting out of machine learning algorithms used to predict if a message is related to disasters.
The data is taken from the "Multilingual Disaster Response Messages" dataset provided by figure eight (https://www.figure-eight.com/dataset/combined-disaster-response-data/)

## Description / Instructions:
The project consists out of three parts.
1. An ETL (Extract-Transform-Load) pipeline, which takes two datasets (messages and categories), cleans them and stores them into a SQLite database.

    - To run ETL pipeline that cleans data and stores in database
    - `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. A ML (Machine-Learning) pipeline, which loads the database and trains a machine learning model using a pipeline. The model is then saved

    - To run ML pipeline that trains classifier and saves
    - `python models/train_classifier.py data/DisasterResponse.db models/model.pck`

3. A web application, which let's you enter own messages. The loaded model will then predict the relation and categories of the message

    - Run the following command in the app's directory to run your web app.
    - `python run.py`
    - When the web app is running, navigate to http://127.0.0.1:3001

### Repository structure

* /app - code for the web application
    * run.py - python file creating a webserver loading the DB and model (see below)
    * /templates
        * go.html - Result HTML after prediction
        * master.html - Main HTML for the webserver
* /data - The ETL pipeline building the SQLite database
    * disaster_categories.csv - The categories provided by figure eight
    * disaster_messages.csv - The messages provided by figure eight
    * ETL_Disaster.db - A preloaded database
    * process_data.py - The ETL pipeline reading in the CSV files resulting in the DB file
* /models - The ML pipeline building the model
    * model.pck - A precreated model
    * train_classifier.py - The ML pipeline reading in the db out of /data and creating the model



## Use it in the wild
https//disaster.schatzschloss.de

## Authors
Myself, but feel free to contribute :)

## Acknowledgements
* Udacity
* figure eight
* Many tutorials, and, of course, StackOverflow

# Licensing
The code is published under `GPL3`: https://www.gnu.org/licenses/gpl-3.0.en.html