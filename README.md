# Financial Message Categorization

## AIM
Categorize messages into "money credited" or "money debited" and  extract the amounts mentioned in the messages based on their predicted categories.

## 1. Gather Data Set and Labeling
- Gather a dataset (list of messages) and label the messages as credited or debited, then store the dataset as a CSV file.

```python
import pandas as pd

dataset = [
    {"text": "Money credited from Anurag", "label": "credited"},
    {"text": "debited Rs50 for groceries", "label": "debited"},
    {"text": "Received rs100 from xyz", "label": "credited"},
    {"text": "ATM withdrawal INR30", "label": "debited"}]

df = pd.DataFrame(dataset)
df.to_csv('financial_dataset.csv', index=False)
```
```
Output example of a csv file

text,label
Money credited from Anurag,credited
debited Rs50 for groceries,debited
Received Rs100 from xyz,credited
ATM withdrawal INR30,debited
```

## 2. Preprocess the data
- Remove irrelevant characters, punctuation, and unnecessary white spaces.
- Convert text to lowercase to ensure consistency.
- Break down the text into individual words or tokens.
- Libraries like NLTK (Natural Language Toolkit) or spaCy can be useful for tokenization.
- Eliminate common words(stopwords), (e.g., "and," "the," "is").
- Reduce words to their base(stemming), (e.g., "running" to "run").
- NLTK or spaCy provide lists of stopwords, stemming and lemmatization functions.
- Identify and extract numerical values from messages using techniques like NER.

### Example (using NLTK library):

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

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

text = "A message about a credited transaction."
processed_text = preprocess_text(text)
print(processed_text)

# Example storage after cleaning

import pandas as pd

# storing the preprocessed data in a CSV file
data = {'text': [processed_text1, processed_text2, ...],
        'label': ['credited', 'debited', ...]}
df = pd.DataFrame(data)
df.to_csv('processed_dataset.csv', index=False)

```

## 3. Train an NLP Model using tensorflow
(You can also use pytorch for doing the same)

- Tokenization and Padding
- here Tokenization means: if the a word in new message, is not know by our model then we should handle it instead of ignoring it.

```python
# Import necessary modules from TensorFlow Keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Assuming 'df' is your DataFrame loaded from the CSV file
# Extract the 'text' and 'label' columns from the DataFrame
texts = df['text'].tolist()
labels = df['label'].tolist()

# Create a Tokenizer with an out-of-vocabulary (OOV) token
tokenizer = Tokenizer(oov_token='<OOV>')
# Fit the tokenizer on the text data to build the vocabulary
tokenizer.fit_on_texts(texts)

# Convert the text data to sequences of integers using the tokenizer
sequences = tokenizer.texts_to_sequences(texts)
# Pad the sequences to ensure uniform length for neural network input
# 'post' padding is used, meaning zeros are added at the end of each sequence to ensure fixedlength
padded_sequences = pad_sequences(sequences, padding='post')

# Calculate the number of unique classes in the 'labels' list
# (predicting "credited" or "debited"), num_classes will be 2.
num_classes = len(set(labels))
```
### Explaination
- 'OOV' stands for "Out-Of-Vocabulary.
- When you set oov_token='<OOV>' during the creation of the Tokenizer, it means that any word not present in the vocabulary (words that were not encountered during the `fit_on_texts` step) will be replaced with the specified out-of-vocabulary token '<OOV>'.
- `fir_on_texts` builds an internal vocabulary index. This vocabulary index maps words to unique integer indices.
- `texts_to_sequences` convert a list of texts (sentences or phrases) into sequences of integers

## 4. Build and Train the Model:
-  using an embedding layer, LSTM layer, and output layer for binary classification.

```python
# Import necessary modules from TensorFlow Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Create a Sequential model which is linear stack of layers
model = Sequential([
    # Embedding layer for word embeddings
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=len(padded_sequences[0])),
    
    # LSTM layer for processing sequential data
    LSTM(64),
    
    # Dense output layer for classification
    Dense(num_classes, activation='softmax')
])

# Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the padded sequences with corresponding labels for 10 epochs
model.fit(padded_sequences, labels, epochs=10)
```
### Explaination
```
# Embedding Layer: 
	- embedding is a representation of words (or tokens) in a continuous vector space.
	- It is used to map each word index to a 32-dimensional vector.
	- 32-dimensional vector, it means that each word in the vocabulary is associated with a vector of 32 numerical values. Each value in the vector is a dimension
	- input_dim is set to the size of the vocabulary (plus one because indexing starts from 1).
	- output_dim is the size of the embedding vector.
	- input_length is the length of input sequences (padded sequences).

# LSTM Layer: 
	- is a type of recurrent neural network (RNN) 
	- Long Short-Term Memory (LSTM)
	- LSTMs are capable of learning long-term dependencies in sequential data, making them well-suited for tasks involving sequences
	- 64 is the number of memory units (or cells) in the LSTM layer.
	- Each LSTM cell maintains its internal state, allowing it to capture different patterns and dependencies in the sequential data.
	- memory cell, which is designed to store information over long periods of time
	- Each memory unit operates independently and contributes to the overall capacity of the LSTM to capture patterns and dependencies in sequential data.

# Dense Output Layer:
	- dense" refers to the fact that every unit in the layer is connected to every unit in the previous layer.
	- it is designed to produce probabilities for each class in a classification task. The model's final prediction is often the class with the highest probability. This layer is suitable for multi-class classification problems, and the use of the softmax activation function ensures that the output is a valid probability distribution over the classes.

# Compile the Model:
	- optimizer='adam': This specifies the optimization algorithm to be used during training. 
	- 'adam' refers to the Adam optimization algorithm, which is a popular and effective choice for a wide range of tasks. 
	- This is the loss function used to measure how well the model is performing.
	- classify tasks where the labels are integers (like 0, 1, 2).
	- sparse_categorical_crossentropy' is a suitable choice. It computes the cross-entropy loss between the true labels and the predicted probabilities.
	means it predict the class of an input sample. The model outputs a probability distribution over all possible classes for each sample.
	- metrics=['accuracy']: This is a list of metrics used to monitor the model's performance during training. 'accuracy' is a commonly used metric for classification problems. It 	represents the proportion of correctly classified samples.

# Model Training:
	- Train the model on the padded sequences (padded_sequences) with corresponding labels (labels) for 10 epochs.
	- 10 epochs means that the model will go through the entire training dataset 10 times, adjusting its weights to improve its ability to make accurate predictions on the training data
```
## 5. Predict whether a new message is related to money being credited or debited :

```python
# Assuming 'new_texts' is a list of new messages
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded_sequences = pad_sequences(new_sequences, padding='post')

predictions = model.predict(new_padded_sequences)
predicted_labels = [1 if pred[0] > 0.5 else 0 for pred in predictions]
```
### Explaination
- The predict method takes the input data and returns the model's predicted probabilities for each class.
- pred[0] represents the predicted probability for the positive class.
-  "positive class," it means messages that the model is predicting as "credited."

## 6. Post Processing predicted_labels:
- extracting amounts, aggregating the amounts mentioned in the messages based on their predicted categories.

```python
# Post-processing: Aggregate amounts for 'credited' and 'debited'
credited_amounts = 0
debited_amounts = 0

for i, label in enumerate(predicted_labels):
    if label == 1:  # 'credited'
        # Extract and aggregate credited amounts (assuming amounts are in the format '$xxx')
        amount_matches = re.findall(r'\$\d+', new_texts[i])
        credited_amounts += sum([float(match[1:]) for match in amount_matches])

    elif label == 0:  # 'debited'
        # Extract and aggregate debited amounts (assuming amounts are in the format '$xxx')
        amount_matches = re.findall(r'\$\d+', new_texts[i])
        debited_amounts += sum([float(match[1:]) for match in amount_matches])

print(f'Total Credited Amounts: ${credited_amounts}')
print(f'Total Debited Amounts: ${debited_amounts}')
```
