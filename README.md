# **Capstone Project - NLP Applications**
## Amazon Product Reviews Sentiment Analysis
This project is a Python script that uses spaCy and TextBlob to analyze the sentiment of Amazon product reviews. It takes a CSV file of product reviews as input and returns the sentiment polarity and label for each review. The sentiment polarity is a numerical value ranging from -1 (negative) to 1 (positive), and the sentiment label is a categorical value of either "Positive", "Negative", or "Neutral".

The purpose of this project is to demonstrate how to use natural language processing (NLP) tools and techniques to perform sentiment analysis on text data. Sentiment analysis is a useful technique for understanding the opinions and feelings of customers, users, or audiences. It can help businesses and organizations improve their products, services, and marketing strategies by identifying the strengths and weaknesses of their offerings, as well as the needs and preferences of their customers.

## How to install and run the project
To install and run the project, you need to have Python 3 and pip installed on your system. You also need to install the following dependencies:

• spaCy: a library for advanced NLP in Python. To install spaCy, run the following command in your terminal: `pip install spacy`

• TextBlob: a library for text processing and sentiment analysis. To install TextBlob, run the following command in your terminal: `pip install textblob`

• spacytextblob: a spaCy pipeline component for TextBlob. To install spacytextblob, run the following command in your terminal: `pip install spacytextblob`

• pandas: a library for data analysis and manipulation. To install pandas, run the following command in your terminal: `pip install pandas`

After installing the dependencies, you can download the project files from the GitHub repository:

`git clone https://github.com/Thenmozhi-Arumugam/finalCapstone.git`

To run the project, navigate to the project directory and execute the Python script:

`cd finalCapstone python sentiment_analysis.py`

The script will ask you to enter the path of the CSV file that contains the product reviews. The CSV file should have a column named "reviews.text" that contains the text of the reviews. You can use the sample file "amazon_product_reviews.csv" from https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products, or you can use your own file.

The script will then process the reviews and print the sentiment polarity and label for each review, as well as the average sentiment polarity and label for all the reviews. You can also view the results in a pandas data frame by uncommenting the last line of the script.

## Idea of the project
This is Task 21 - Capstone Project - NLP Applications from CoGrammar/HyperionDev Bootcamp.

## The resources
The resources for the task used can be found in the attached PDF file, and there are described preprocessing steps and model.
