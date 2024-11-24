import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained('./fine_tuned_model', 
                                                          local_files_only=True)  # Ensure we load locally
    tokenizer = BertTokenizer.from_pretrained('./fine_tuned_model', 
                                              local_files_only=True)  # Ensure we load locally
    return model, tokenizer

# Function to predict sentiment
def predict_sentiment(text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get prediction
    prediction = torch.argmax(logits, dim=1).item()
    
    # Map prediction to sentiment
    if prediction == 1:
        return "Positive"
    else:
        return "Negative"

# Streamlit App
def main():
    st.title("Movie Review Sentiment Analysis")
    st.write("Enter a movie review, and the model will predict whether it is positive or negative.")

    # Input for the review text
    review_text = st.text_area("Enter the movie review:")

    if st.button("Analyze Sentiment"):
        if review_text:
            # Load the model and tokenizer
            model, tokenizer = load_model()

            # Predict the sentiment
            sentiment = predict_sentiment(review_text, model, tokenizer)

            # Display the result
            st.write(f"Sentiment: {sentiment}")
        else:
            st.warning("Please enter a review to analyze.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
