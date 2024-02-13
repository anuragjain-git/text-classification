import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

import pickle


nltk.download('stopwords')
nltk.download('punkt')

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

def preprocess_text_list(text_list):
    preprocessed_texts = [preprocess_text(text) for text in text_list]
    return preprocessed_texts

texts = [
    # Axis Bank
    "Debit INR 500.00 A/c no. XX8926 12-10-23 20:02:19 UPI/P2A/328546155288/ANURAG JAIN SMS BLOCKUPI Cust ID to 01351860002, if not you - Axis Bank",
    "Debit INR 109.00 A/c no. XX8926 27-01-24 11:36:57 UPI/P2M/6321837696198/Add Money to Wallet SMS BLOCKUPI Cust ID to 919951860002, if not you - Axis Bank",
    "INR 5590.00 credited to A/c no. XX8926 on 09-11-23 at 11:59:28 IST. Info- UPI/P2A/334365332111/ANURAG JAIN/Axis Bank - Axis Bank",
    "INR 216.35 credited to A/c no. XX8926 on 06-01-24 at 07:32:16 IST. Info- NEFT/CMS333334641/NEXTBIL. Avl Bal- INR 33478.22 - Axis Bank",
    # UCO Bank
    "A/c XX8360 Debited with Rs. 19.00 on 07-02-2024 by UCO-UPI.Avl Bal Rs.32.98. Report Dispute https://bit.ly/3y39tLP .For feedback https://rb.gy/fdfmda",
    "A/c XX8360 Credited with Rs.6.00 on 07-02-2024 by UCO-UPI.Avl Bal Rs.51.98. Report Dispute https://bit.ly/3y39tLP .For feedback https://rb.gy/fdfmda",
    # SBI
    "Dear UPI user A/C X0429 debited by 20.0 on date 22Jan24 trf to Mr Narayan Badat Refno 437652379634. If not u? call 1800111109. -SBI",
    "Dear SBI UPI User, ur A/cX0429 credited by Rs500 on 04Feb24 by  (Ref no 403585759002)",
    # Union Bank
    "A/c *9172 Debited for Rs:50.00 on 11-02-2024 19:44:40 by Mob Bk ref no 444816787760 Avl Bal Rs:1870.55.If not you, Call 1800222243 -Union Bank of India",
    "A/c *9172 Credited for Rs:501.00 on 23-01-2024 20:05:45 by Mob Bk ref no 402347890661 Avl Bal Rs:556.00.Never Share OTP/PIN/CVV-Union Bank of India",
    # Federal Bank
    "Rs 50.00 debited from your A/c using UPI on 03-02-2024 16:44:28 to VPA abcd4321@oksbi - (UPI Ref No 403417856009)-Federal Bank",
    # Kotak Bank
    "Sent Rs.20.00 from Kotak Bank AC X8136 to abcd2003@oksbi on 03-02-24.UPI Ref 403418725300. Not you, kotak.com/fraud",
    "Received Rs.50.00 in your Kotak Bank AC X8136 from abcd4321@oksbi on 03-02-24.UPI Ref:400653974000.",
    # HDFC Bank
    "UPDATE: INR 1,000.00 debited from HDFC Bank XX2002 on 11-DEC-23. Info: FT - Dr - XXXXXXXXXX1498 - ANURAG JAIN. Avl bal:INR 4,891.00",
    "HDFC Bank: Rs. 1.00 credited to a/c XXXXXX2002 on 23-01-24 by a/c linked to VPA 9777777711@fam (UPI Ref No 408888887329).",
    # Jio Payments Bank
    "Your JPB A/c xxxx0956 is credited with Rs.25.00 on 25-Aug-2023. Your current  account balance is Rs.25.",
    # Paytm Payments Bank
    "Rs.550 sent to abcd1234-1@okicici from PPBL a/c 91XX8089.UPI Ref:439432479819;Balance:https://m.paytm.me/pbCheckBal; Help:http://m.p-y.tm/care",
    # Extra
    "IRCTC CF has requested money on Google Pay UPI app. On approving, INR 1033.60 will be debited from your A/c - Axis Bank",
    "You have received UPI mandate collect request from TATA TECHNOLOGIES LI for INR 15000.00. Log into Google Pay app to authorize - Axis Bank",
    "ANURAG JAIN has requested money from you on Google Pay. On approving the request, INR 31.00 will be debited from your A/c - Axis Bank",
    "Flipkart Refund Processed: Refund of Rs. 237.0 for favoru Household wrap ... is successfully transferred and will be credited to your account by Oct 04, 2023.",
    "UPI mandate has been successfully created towards TATA TECHNOLOGIES LI for INR 15000.00. Funds blocked from A/c no. XX8926. 12e5d61d2ac145738241fbf117bb295c@okaxis - Axis Bank",
    "hi"
]

# Preprocess the texts
processed_texts = preprocess_text_list(texts)

# Example storage after cleaning
data = {'text': processed_texts,
        'label': ['debited', 'debited', 'credited', 'credited', 'debited', 'credited', 'debited', 'credited', 'debited', 'credited', 'debited', 'debited', 'credited', 'debited', 'credited', 'credited','debited', 'requested', 'requested', 'requested', 'willcredit', 'blocked', 'x']}
df = pd.DataFrame(data)
df.to_csv('processed_dataset.csv', index=False)

# Load the processed dataset from the CSV file
df = pd.read_csv('processed_dataset.csv')

# Extract the 'text' and 'label' columns from the DataFrame
texts = df['text'].tolist()
labels = df['label'].tolist()

# Create a Tokenizer with an out-of-vocabulary (OOV) token
tokenizer = Tokenizer(oov_token='<OOV>')
print(tokenizer)
tokenizer.fit_on_texts(texts)

# Save the tokenizer to a file
with open('tokenizer.pkl', 'wb') as token_file:
    pickle.dump(tokenizer, token_file)

# Convert the text data to sequences of integers using the tokenizer
sequences = tokenizer.texts_to_sequences(texts)
print(sequences)
# Pad the sequences to ensure uniform length for neural network input
padded_sequences = pad_sequences(sequences, padding='post')

# Calculate the number of unique classes in the 'labels' list
num_classes = len(set(labels))

# Create a Sequential model
model = Sequential([
    # Embedding layer for word embeddings
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32),
    
    # LSTM layer for processing sequential data
    LSTM(100),
    
    # Dense output layer for classification
    Dense(num_classes, activation='softmax')
])

# Assuming 'df' is your DataFrame containing the 'label' column
label_encoder = LabelEncoder()
df['encoded_label'] = label_encoder.fit_transform(df['label'])

# Extract the encoded labels
encoded_labels = df['encoded_label'].tolist()

# Convert labels to NumPy array
labels_np = np.array(encoded_labels)
# Replace the lambda function with a named function
def custom_sparse_softmax_cross_entropy(labels, logits):
    return tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, logits)

# Compile the model with the named function
model.compile(optimizer='adam', loss=custom_sparse_softmax_cross_entropy, metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels_np, epochs=100)
# Save the model in the recommended Keras format
model.save('trained_model.keras')
