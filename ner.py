import spacy
from spacy.tokens import DocBin
from spacy.training import Example

# Load the spaCy English model
nlp = spacy.blank("en")

# Define training data with labeled examples
TRAIN_DATA = [
    ("Dear SBI UPI User, ur A/cX0304 debited by Rs91000 on 08Feb24 by (Ref no 403968023837)", {"entities": [(14, 20, "AC_NUMBER"), (29, 35, "AMOUNT"), (45, 53, "DATE")]}),
    # Add more labeled examples here
]

# Prepare training examples
examples = []
for text, annot in TRAIN_DATA:
    examples.append(Example.from_dict(nlp.make_doc(text), annot))

# Initialize the pipeline components
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Add entity labels to the NER pipeline
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Disable other pipeline components for training
pipe_exceptions = ["ner"]
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# Train the NER model
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for itn in range(10):  # Adjust number of iterations
        losses = {}
        examples = spacy.util.minibatch(examples, size=4)  # Adjust batch size
        for batch in examples:
            nlp.update(batch, drop=0.5, losses=losses)

# Save the trained model
nlp.to_disk("trained_ner_model")

# Load the trained model
nlp = spacy.load("trained_ner_model")

# Test the trained model
test_text = "Dear SBI UPI User, ur A/cX0304 debited by Rs91000 on 08Feb24 by (Ref no 403968023837)"
doc = nlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
