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
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Initialize an empty list to store processed characters
    processed_chars = []
    
    i = 0
    while i < len(text):
        # If character is a digit, skip all characters until the next space
        if text[i].isdigit():
            while i < len(text) and text[i] != ' ':
                i += 1
        # If character is alphanumeric or space, add it to processed_chars
        elif text[i].isalnum() and not text[i].isdigit() or text[i].isspace():
            processed_chars.append(text[i])
        i += 1
    
    # Join the processed characters into a string
    processed_text = ''.join(processed_chars)
    
    # Tokenization
    tokens = word_tokenize(processed_text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

# Example usage:
text = "This is an example text with some numbers like 12345, email like abc@gmail.com and punctuation! But we'll remove them."
processed_text = preprocess_text(text)
print(processed_text)


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
    "UPI mandate has been successfully created towards TATA TECHNOLOGIES LI for INR 15000.00. Funds blocked from A/c no. XX8926. 12e5d61d2ac145738241fbf117bb295c@okaxis - Axis Bank"
]

# Preprocess the texts
processed_texts = preprocess_text_list(texts)

# Example storage after cleaning
data = {'text': processed_texts,
        'label': ['debited', 'debited', 'credited', 'credited', 'debited', 'credited', 'debited', 'credited', 'debited', 'credited', 'debited', 'debited', 'credited', 'debited', 'credited', 'credited','debited', 'requested', 'requested', 'requested', 'willcredit', 'blocked']}
df = pd.DataFrame(data)
df.to_csv('processed_dataset.csv', index=False)

# Load the processed dataset from the CSV file
df = pd.read_csv('processed_dataset.csv')

# Extract the 'text' and 'label' columns from the DataFrame
texts = df['text'].tolist()
labels = df['label'].tolist()

# Create a Tokenizer with an out-of-vocabulary (OOV) token
# this will replace any unknown words with a token of our choosing
tokenizer = Tokenizer(num_words=95000, oov_token='OOV', filters='!"#$%&()*+,-./:;<=>@[\]^_`{|}~ ')
tokenizer.fit_on_texts(texts)

# Save the tokenizer to a file
with open('tokenizer.pkl', 'wb') as token_file:
    pickle.dump(tokenizer, token_file)

# Convert the text data to sequences of integers using the tokenizer
sequences = tokenizer.texts_to_sequences(texts)

# Pad the sequences to ensure uniform length for neural network input
padded_sequences = pad_sequences(sequences, padding='post')

# Calculate the number of unique classes in the 'labels' list
num_classes = len(set(labels))

# Create a Sequential model
model = Sequential([
    # Embedding layer for word embeddings
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32),
    
    # LSTM layer for processing sequential data
    LSTM(50),
    
    # Dense output layer for classification
    Dense(num_classes, activation='softmax')
])

# Assuming 'df' is your DataFrame containing the 'label' column
label_encoder = LabelEncoder() # will be used to convert categorical labels into numerical labels.
df['encoded_label'] = label_encoder.fit_transform(df['label']) # transform these labels into numerical format

# Extract the encoded labels
encoded_labels = df['encoded_label'].tolist()

# Convert labels to NumPy array
labels_np = np.array(encoded_labels)

# Replace the lambda function with a named function
def custom_sparse_softmax_cross_entropy(labels, logits):
    return tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

# Compile the model with the named function
model.compile(optimizer='adam', loss=custom_sparse_softmax_cross_entropy, metrics=['accuracy', 'precision', 'recall'])

# Train the model
# model.fit(padded_sequences, labels_np, epochs=100)

# Assuming you have stored the model training history in a variable named 'history'
history = model.fit(padded_sequences, labels_np, epochs=100, validation_split=0.2)

# Extracting training and validation loss from history
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Plotting the training and validation loss
epochs = range(1, len(training_loss) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_loss, 'bo-', label='Training Loss')
plt.plot(epochs, validation_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Save the model in the recommended Keras format
model.save('trained_model.keras')