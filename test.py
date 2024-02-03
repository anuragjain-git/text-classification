import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder

# Load your custom dataset from CSV
df = pd.read_csv('processed_dataset.csv')  # Make sure your CSV file is correctly formatted

# Set a maximum sequence length for your tokenizer
max_seq_length = 50  # Adjust as needed

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenize and prepare the dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CustomDataset(Dataset):
    def __init__(self, texts, labels, label_encoder, max_seq_length):
        self.texts = texts
        self.labels = label_encoder.transform(labels)
        self.label_encoder = label_encoder
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = tokenizer(
            self.texts[idx],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length
        )
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Convert to tensor with type long

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': label
        }

    def collate_fn(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['label'] for item in batch]

        # Pad sequences to the length of the longest sequence in the batch
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        padded_attention_masks = pad_sequence(attention_masks, batch_first=True)

        return {'input_ids': padded_input_ids, 'attention_mask': padded_attention_masks}, torch.stack(labels)

# Create label encoder
label_encoder = LabelEncoder()
label_encoder.fit(df['label'])

# Create DataLoader for training and validation sets
train_dataset = CustomDataset(train_df['text'].tolist(), train_df['label'].tolist(), label_encoder, max_seq_length)
val_dataset = CustomDataset(val_df['text'].tolist(), val_df['label'].tolist(), label_encoder, max_seq_length)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=train_dataset.collate_fn)


# Fine-tune the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['label'].unique()))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        inputs, labels = batch
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save the fine-tuned model
model.save_pretrained(r"C:\Users\Anurag\OneDrive\Desktop\aimodel")

# Use the fine-tuned model for zero-shot classification
model = BertForSequenceClassification.from_pretrained(r"C:\Users\Anurag\OneDrive\Desktop\aimodel")

# Extract scores and labels
def get_scores(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    scores = torch.sigmoid(logits).detach().numpy().flatten().tolist()
    return scores

# Example zero-shot classification
text_to_classify = "debit inr 50"
candidate_labels = ['credited', 'debited', 'requested']

result = get_scores(text_to_classify)
result_dict = dict(zip(candidate_labels, result))

print(result_dict)
