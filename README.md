# Disaster-Response-Classification

This project deals with classifying messages received during a disaster into their respective categories. The data set contains disaster messages along with their categories, genre, and ID. 

### Description:



### Required libraries to install:
Python 3.x version along with the following libraries are required to run the files,
NumPy 
Pandas
Pickle
Sys
Re
Scikit-Learn
NLTK
Matplotlib
SQLalchemy
Json
Flask
Plotly

### Instructions to Run:
To clone this repository, execute the following command in the terminal,
git clone https://github.com/Shravanthi98/Disaster-Response-Classification.git

After setting up the project folder, follow the instructions below to run the code,
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans the data and stores it in a database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Disaster.db`
    - To run ML pipeline that trains a classifier and saves it
        `python models/train_classifier.py data/Disaster.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ to see the app up and running. Enter a message (asking for help, info, warning, etc.) in English as an input and the model outputs all the categories relevant to the given input.

### Licensing, Authors, Acknowledgements:
No License. 
Data: Figure Eight.
