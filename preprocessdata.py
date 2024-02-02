import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text_list(text_list):
    preprocessed_texts = []
    for text in text_list:
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
        
        preprocessed_texts.append(' '.join(tokens))
    
    return preprocessed_texts

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
processed_texts = preprocess_text_list(texts)

for processed_text in processed_texts:
    print(processed_text)

# Example storage after cleaning

import pandas as pd

# storing the preprocessed data in a CSV file
data = {'text': processed_texts,
        'label': ['debited','debited','credited','credited','credited','requested','requested','requested','willCredited','debited']}
df = pd.DataFrame(data)
df.to_csv('processed_dataset.csv', index=False)

