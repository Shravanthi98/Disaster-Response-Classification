# Disaster-Response-Classification

### Motivation:
This project deals with classifying messages received during a disaster into their respective categories. The model is intended to categorize these disaster messages to avail the right kind of help during an emergency by directing it to an appropriate disaster relief agency.

### Description:
The data set contains disaster messages along with their categories, original message, genre, and ID. 
The project is divided into three stages,
* ETL Pipeline: 
  In this stage, the data is extracted from the csv files and transformed by necessary cleaning and processing followed by loading into a SQL data base.
  "process_data.py" runs this pipeline.
* Machine Learning (ML) Pipeline:
  In this part, features are extracted from the processed text data which are then used to train a Multi-output Ada Boost classifier that learns to classify these messages     into their respective categories. The trained model is evaluated on a bunch of metrics like Accuracy, Precision, Recall, and F1-score for each of the categories/labels.
  "train_classifier.py" runs this pipeline.
* Deploying it as a web-app:
  The model is then deployed using flask and visualizations are rendered using Plotly. 

### Required libraries to install:
Python 3.x version along with the following libraries are required to run the files,
1. NumPy 
2. Pandas
3. Pickle
4. Sys
5. Re
6. Scikit-Learn
7. NLTK
8. Matplotlib
9. SQLalchemy
10. Json
11. Flask
12. Plotly

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
