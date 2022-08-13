import re
from settings import TEXT_PREPROCESSING
if TEXT_PREPROCESSING == "lemma":
    import spacy
    nlp = spacy.load('de_core_news_lg', disable=['ner', 'parser'])  # also possible: 'de_core_news_md'

elif TEXT_PREPROCESSING == "stem":
    from nltk.stem import SnowballStemmer
    stem = SnowballStemmer('german')


def text_preprocessing(input_series):
    """Text preprocessing in a sklearn-pipeline:
    The function takes a series of inputs and outputs a preprocessed series (in the same order!)
    The following transformations are applied:
        - stripping the text
        - replace URLs and emails with the word itself
        - remove special characters
        - remove digits
        - optional: lemmatize or stem the text
    """

    output_series = []
    for textinput in input_series:
        textinput = str(textinput)  # make sure, the textinput is type string

        # replace email with the word email
        textinput = re.sub(r"([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)", r"email", textinput)

        # replace url with the word url
        textinput = re.sub(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", "url", textinput)

        # remove special characters and newlines
        textinput = re.sub(r"[^a-zA-ZäöüßÄÖÜ€$ ]", r" ", textinput)

        if TEXT_PREPROCESSING is None:
            # replace newlines and multiple whitespaces with a single whitespace
            textinput = " ".join(textinput.split())

        elif TEXT_PREPROCESSING == "lemma":  # (recommended)
            # check if word exists in the corpus and if so replace it with its lemma
            # please note: the SpaCy corpus does not cover all german words, but it's still recommended
            # since the PDF text layer contains often gibberish words
            textinput = " ".join([token.lemma_ for token in nlp(textinput) if nlp.vocab.has_vector(token.text)])

        elif TEXT_PREPROCESSING == "stem":
            # replace newlines and multiple whitespaces with a single whitespace
            textinput = " ".join(textinput.split())
            textinput = " ".join(stem.stem(word) for word in str(textinput).split() if len(word) > 2)

        # further preprocessing:
        # Sometimes, the documents contain misleading data (such as names) which you might want to delete:
        # You can do so by coding: textinput = textinput.replace(NAME, "") or, if you have multiple deletions:
        # delete_list = [NAME, PLACE, DATE]
        # for deletion in delete_list:
        #     textinput = textinput.replace(deletion, "")
        # please use a dictionary, if you want to replace words instead of deleting them.

        output_series.append(textinput)

    return output_series
