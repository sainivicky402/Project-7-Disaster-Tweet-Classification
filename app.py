import streamlit as st
import pickle

# Load the model and vectorizer
with open('disaster_tweet_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Set the title of the app
st.title("Disaster Tweet Classifier")

# Create a text input for the user to enter a tweet
tweet = st.text_area("Enter the tweet text:")

# Button to make a prediction
if st.button("Predict"):
    if tweet:
        # Transform the input using the vectorizer
        tweet_tfidf = vectorizer.transform([tweet])
        # Predict using the loaded model
        prediction = model.predict(tweet_tfidf)

        # Display the prediction result
        if prediction[0] == 1:
            st.success("This tweet is related to a disaster.")
        else:
            st.success("This tweet is not related to a disaster.")
    else:
        st.warning("Please enter a tweet to classify.")
