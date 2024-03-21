# Importing spacy and pandas.
import spacy
import pandas as pd
from textblob import TextBlob

# Specifying the model we want to use.
nlp = spacy.load('en_core_web_md')

# Making data frame from csv file.
df = pd.read_csv('amazon_product_reviews.csv')

# Remove all missing values.
reviews_data = df.dropna(subset=['reviews.text'])

# Function to clean data
def clean_reviews(reviews):
    """ 
    Tokenize reviews and remove stop words
    Methods suggested in pdf instructions
     Parameters: 
    - reviews.text (str):product reviews from amazon_product_reviews.csv
    
    Returns: 
    - ready_reviews (str): cleaned product reviews
    """
    doc = nlp(reviews)
    ready_reviews = [token.text for token in doc if not token.is_stop]
    
    # Formatting the text to lowercase using lower() and removing additional spaces using strip()
    ready_reviews = ' '. join(ready_reviews).lower().strip()

    return ready_reviews

# Apply the clean_reviews function to the reviews text in the df.
# Save as a new ready_reviews column in the df.
reviews_data['ready_reviews'] = reviews_data['reviews.text'].apply(clean_reviews)

# Print first ten reviews of cleaned reviews to check if everything is working
print(reviews_data['ready_reviews'].head(10))

# Function for Sentiment Analysis
def predict_sentiment(ready_reviews):
    """
    Two-part function to analyse sentiment in cleaned customer reviews of products.
    Incorporating .sentiment and .polarity attributes.

    Parameters: 
    - ready_reviews (str):cleaned product reviews from 'reviews.text' in amazon_product_reviews.csv
    
    Returns: 
    - sentiment (str): predicted sentiment positive, negative or neutral sentiment
    - polartiy (float): predicted sentiment -1 to 1, -1 being negative and 1 being positive
    """
    # Create TextBlob object
    blob = TextBlob(ready_reviews)
    # Using the polarity attribute
    polarity = blob.polarity
    # Using the sentiment attribute
    sentiment = blob.sentiment
    # Return the sentiment score and label.
    if polarity > 0.15:
          sentiment_label = "Positive"
          return sentiment, sentiment_label
    elif polarity > 0:
          sentiment_label = "Neutral"
          return sentiment, sentiment_label
    else:
          sentiment_label = "Negative"
          return sentiment, sentiment_label
    
# Test and print model on five product reviews with index for easy referencing in other uses.
for i, review in enumerate(reviews_data['ready_reviews'].head(5)):
    sentiment, sentiment_label = predict_sentiment(review)
    # print sentiment (polarity and subjectivity score) and sentiment_label for the 5 sample reviews
    print(f"Customer review {i + 1} - {sentiment}, {sentiment_label}")

# Calculate similarity between reviews
print("----- Similarity -----")

# Select specific reviews for comparison
my_review_of_choice = reviews_data['ready_reviews'][5]
my_second_review_of_choice = reviews_data['ready_reviews'][150]


def calculate_chosen_similarity(my_review_of_choice, my_second_review_of_choice):
    """
    Measures semantic similarity between two chosen cleaned reviews. 
    Returns similarity score to help us understand deviation in customer opinion.
    
    Parameters:
    - my_review_of_choice(list): one item list containing a randomly selected cleaned review
    - my_second_review_of_choice(list): one item list containing a randomly selected cleaned review

    Returns:
    - Similarity score
    """

    #Process the review texts
    chosen_review_1 = nlp(my_review_of_choice)
    chosen_review_2 = nlp(my_second_review_of_choice)

    #Calculate similarity
    chosen_similarity_score = chosen_review_1.similarity(chosen_review_2)

    return chosen_similarity_score


# Apply to randomly selected reviews
chosen_similarity_score = calculate_chosen_similarity(my_review_of_choice, my_second_review_of_choice)
# Round up similarity
rounded_chosen_similarity_score = round(chosen_similarity_score, 2)
# Print orginial comments and similarity score

print(f"Similarity between customer reviews 5 and 150 is {rounded_chosen_similarity_score}.")