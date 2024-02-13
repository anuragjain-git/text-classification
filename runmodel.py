import tensorflow as tf
from keras.models import load_model
from model import preprocess_text  # Import your preprocessing function
import pickle
from model import processed_texts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


# Define the custom loss function
def custom_sparse_softmax_cross_entropy(labels, logits):
    return tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, logits)

# Load the trained model using the custom loss function
loaded_model = load_model('trained_model.keras', custom_objects={'custom_sparse_softmax_cross_entropy': custom_sparse_softmax_cross_entropy})

df = pd.read_csv('processed_dataset.csv')

# Load the processed_texts list
# with open('processed_texts.pkl', 'rb') as f:
#     processed_texts = pickle.load(f)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as token_file:
    tokenizer = pickle.load(token_file)

label_encoder = LabelEncoder()

# Assuming 'new_texts' is a list of new messages
new_texts = [
    "Dear Player, Rs.10,000* is credited to your RummyTime a/c Ref Id: RT210XX Download the app & make your Ist deposit now - http://gmg.im/bKSfAL T&C Apply",
    "UPI Bank account is credited with RS.25.00 on 25-Aug-2023",
    "credit INR refund 100",
    "Refund Processed: Refund of Rs. 237.0 for favoru Household wrap ... is successfully transferred and will be credited to your account by Oct 04, 2023.", 
    "UPI mandate has been successfully created towards TATA TECHNOLOGIES LI for INR 15000.00. Funds blocked from A/c no. XX8926. 12e5d61d2ac145738241fbf117bb295c@okaxis - Axis Bank",
    "Dear Player, Rs.10,000* is credited to your RummyTime a/c Ref Id: RT210XX Download the app & make your 1st deposit now - http://gmg.im/bKSfALT&C Apply"]

similarity_threshold = 0.7

for text in new_texts:
    # Preprocess the new text using spaCy
    preprocessed_new_text = preprocess_text(text)

    # Calculate similarity between the new text and each text in the dataset using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed_new_text] + processed_texts)
    similarity_scores = cosine_similarity(tfidf_matrix)[0][1:]

    # Predictions
    new_sequences = tokenizer.texts_to_sequences([preprocessed_new_text])
    new_padded_sequences = pad_sequences(new_sequences, padding='post')
    predictions = loaded_model.predict(new_padded_sequences)
    predicted_labels = [label for label in predictions.argmax(axis=1)]
        
    # Inverse transform predicted labels to original class labels
    # Ensure that you have fitted the LabelEncoder before transforming
    label_encoder.fit(df['label'])
    predicted_class_labels = label_encoder.inverse_transform(predicted_labels)

    # Check relevance and print the result
    is_relevant = any(score >= similarity_threshold for score in similarity_scores)
    relevance_status = "Relevant" if is_relevant else "Irrelevant"
    print(f"Text: {text} | Predicted Label: {predicted_class_labels[0]} | Relevance: {relevance_status}")

