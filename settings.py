import os
import warnings
import logging
logging.basicConfig(level=logging.CRITICAL)  # set to DEBUG to see if anything is working
logger = logging.getLogger("PyPDF2")
logger.setLevel(logging.ERROR)

BASE_DIR = os.getcwd()
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
PREDICT_DIR = os.path.join(BASE_DIR, "predict")



def get_least_populated_category():
    folderlist = ([x for x in os.walk(DOCUMENTS_DIR)])

    min_file_count = []
    for folder in folderlist:
        i = 0
        for file in folder[2]:
            if str(file).lower().endswith(".docx") or str(file).lower().endswith(".pdf"):
                i += 1
        if i > 0:
            min_file_count.append(i)
    return min(min_file_count)


# see source[3] for the difference between stemming and lemmatization
TEXT_PREPROCESSING = "lemma"  # "lemma" (recommended) for lemmatization; "stem" for stemming and None for none.

MAX_WORDS = 500

CV_SPLITS = min(([int(get_least_populated_category()/2), 15]))  # set the CV_SPLITS to half of the least populated category; max: 15

if CV_SPLITS < 5:
    warnings.warn("Please provide at least 10 examples per category")

EPOCHS = os.environ.get("EPOCHS", "100")  # (100 is recommended), 50 will do fine too. The EPOCHS only apply to the doc2vec training.
EPOCHS = int(EPOCHS)
USE_TRAINED = (os.environ.get('USE_TRAINED', 'False') == 'True')


TRAIN_FINAL = (os.environ.get('TRAIN_FINAL', 'False') == 'True')
TRAIN_ONCE = (os.environ.get('TRAIN_ONCE', 'False') == 'True')

SAVE_TO_DISK = (os.environ.get('SAVE_TO_DISK', 'True') == 'True')

OUTPUT_PROBA = (os.environ.get('OUTPUT_PROBA', 'False') == 'True')  # set to True, if you're interested in the probabilities

CUSTOM_NAME_SUFFIX = os.environ.get('CUSTOM_NAME_SUFFIX', '')

MODEL = os.environ.get('MODEL', 'Doc2Vec')
