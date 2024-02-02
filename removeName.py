import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

def replace_person_names(text):
    words = word_tokenize(text)
    tagged = pos_tag(words)
    entities = ne_chunk(tagged)

    # Replace person names with "PERSON"
    replaced_text = [word[0] if isinstance(word, tuple) and word[1] != 'NNP' else 'PERSON' if isinstance(word, Tree) and word.label() == 'PERSON' else str(word) for word in entities]

    return ' '.join(replaced_text)

text = "A message about a credited transaction. John Smith sent the payment."
processed_text = replace_person_names(text)
print(processed_text)
