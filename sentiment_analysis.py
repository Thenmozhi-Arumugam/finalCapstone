# Importing spacy and pandas.
import spacy
import pandas as pd
from nltk.corpus import stopwords

# Add the textblob component to the pipeline.
from spacytextblob.spacytextblob import SpacyTextBlob
# Specifying the model we want to use.
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# Loads stop words to prepare the text to be analyzed
cached_stop_words = stopwords.words("english")

# Making data frame from csv file.
df = pd.read_csv('amazon_product_reviews.csv')

reviews_data = df['reviews.text']

# Remove all missing values.
clean_data = reviews_data.dropna()


def predict_sentiment(review):
   """
   The function that takes a text review and returns its sentiment.

   Parameter
     ---------
     review : str
     Release review in string format.

   Returns
     float
     The sentiment polarity of the review,
     ranging from -1 (negative) to 1 (positive).
     str
     The sentiment label of the review   

   Examples
     predict_sentiment("I love this product!")
     (0.5, 'Positive')
     predict_sentiment("This is the worst thing ever.")
     (-1.0, 'Negative')
        
   """
  
   # Process the review with spaCy.
   doc = nlp(review)

   # Get the sentiment polarity of the document.
   polarity = doc._.blob.polarity

   # Get the sentiment subjectivity of the document.
   subjectivity = doc._.blob.subjectivity
   print(subjectivity)
   # Return the sentiment score and label.
   if polarity > 0.15:
        return polarity, "Positive"
   elif polarity > 0:
        return polarity, "Neutral"
   else:
        return polarity, "Negative"


# Test the function with a sample review.
review = clean_data[5]
print(predict_sentiment(review))
print(clean_data[5])

# Test the function with a sample review.
review = clean_data[15]
print(predict_sentiment(review))
print(clean_data[15])

# Test the function with a sample review.
review = clean_data[93]
print(predict_sentiment(review))
print(clean_data[93])