import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    text = ''.join([char.lower() for char in text if char.isalnum() or char.isspace()])
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

def check_relevance(new_text, dataset_texts, similarity_threshold=0.5):
    # Preprocess the new text
    preprocessed_new_text = preprocess_text(new_text)

    # Preprocess each text in the dataset
    preprocessed_dataset_texts = [preprocess_text(text) for text in dataset_texts]

    # Calculate similarity between the new text and each text in the dataset
    vectorizer = CountVectorizer().fit_transform([preprocessed_new_text] + preprocessed_dataset_texts)
    similarity_matrix = cosine_similarity(vectorizer)

    # Get the similarity scores
    similarity_scores = similarity_matrix[0][1:]

    # Check if any text in the dataset is similar to the new text
    return any(score >= similarity_threshold for score in similarity_scores)

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

# Load the tokenizer
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.word_index = np.load('tokenizer_word_index.npy', allow_pickle=True).item()

# Redefine custom loss function with the same structure as during training
def custom_loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits)

# Load the model with the correct custom loss function
loaded_model = load_model('trained_model.h5', custom_objects={'custom_loss': custom_loss})

texts = [
    "Debit INR 500.00 A/c no. XX8926 12-10-23 20:02:19 UPI/P2A/328546155288/ANURAG JAIN SMS BLOCKUPI Cust ID to 01351860002, if not you - Axis Bank",
    "Debit INR 109.00 A/c no. XX8926 27-01-24 11:36:57 UPI/P2M/6321837696198/Add Money to Wallet SMS BLOCKUPI Cust ID to 919951860002, if not you - Axis Bank",
    "INR 5590.00 credited to A/c no. XX8926 on 09-11-23 at 11:59:28 IST. Info- UPI/P2A/334365332111/ANURAG JAIN/Axis Bank - Axis Bank",
    "INR 216.35 credited to A/c no. XX8926 on 06-01-24 at 07:32:16 IST. Info- NEFT/CMS333334641/NEXTBIL. Avl Bal- INR 33478.22 - Axis Bank",
    "Your JPB A/c xxxx0956 is credited with Rs.25.00 on 25-Aug-2023. Your current  account balance is Rs.25.",
    "IRCTC CF has requested money on Google Pay UPI app. On approving, INR 1033.60 will be debited from your A/c - Axis Bank",
    "You have received UPI mandate collect request from TATA TECHNOLOGIES LI for INR 15000.00. Log into Google Pay app to authorize - Axis Bank",
    "SOURAV CHANDRA DEY has requested money from you on Google Pay. On approving the request, INR 31.00 will be debited from your A/c - Axis Bank",
    "Flipkart Refund Processed: Refund of Rs. 237.0 for favoru Household wrap ... is successfully transferred and will be credited to your account by Oct 04, 2023.",
    "UPI mandate has been successfully created towards TATA TECHNOLOGIES LI for INR 15000.00. Funds blocked from A/c no. XX8926. 12e5d61d2ac145738241fbf117bb295c@okaxis - Axis Bank"
]

# Assuming 'new_texts' is a list of new messages
new_texts = ["debit text here asjkxbsa axbjsa  xjasbx xasgxya yxyagsvxtyasf  61t72t7172 ",
             "credit INR refund 100",
             "Refund Processed: Refund of Rs. 237.0 for favoru Household wrap ... is successfully transferred and will be credited to your account by Oct 04, 2023.",
             "UPI mandate has been successfully created towards TATA TECHNOLOGIES LI for INR 15000.00. Funds blocked from A/c no. XX8926. 12e5d61d2ac145738241fbf117bb295c@okaxis - Axis Bank",
             "Dear Player, Rs.10,000* is credited to your RummyTime a/c Ref Id: RT210XX Download the app & make your 1st deposit now - http://gmg.im/bKSfALT&C Apply"]

# Check relevance and print the result for new texts
for text in new_texts:
    # Preprocess the new text
    preprocessed_new_text = preprocess_text(text)

    # Tokenize and pad the new text
    new_sequences = tokenizer.texts_to_sequences([preprocessed_new_text])
    new_padded_sequences = pad_sequences(new_sequences, padding='post')

    # Predictions
    predictions = loaded_model.predict(new_padded_sequences)
    predicted_labels = [label for label in predictions.argmax(axis=1)]

    # Inverse transform predicted labels to original class labels
    predicted_class_labels = label_encoder.inverse_transform(predicted_labels)

    # Check relevance and print the result
    is_relevant = check_relevance(preprocessed_new_text, texts)
    relevance_status = "Relevant" if is_relevant else "Irrelevant"
    print(f"Text: {text} | Predicted Label: {predicted_class_labels[0]} | Relevance: {relevance_status}")
