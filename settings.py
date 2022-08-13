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

USE_TRAINED = True

def get_least_populated_category():
    folderlist = ([x for x in os.walk(DOCUMENTS_DIR)])

    # ignored (empty) folders are set to 10000, therefore don't affect the minimum
    return len(min(folderlist[1:], key=lambda folder: len(folder[2]) if len(folder[2]) != 0 else 10000)[2])


# see source[3] for the difference between stemming and lemmatization
TEXT_PREPROCESSING = "lemma"  # "lemma" (recommended) for lemmatization; "stem" for stemming and None for none.

MAX_WORDS = 500

CV_SPLITS = min(([int(get_least_populated_category()/2), 15]))  # set the CV_SPLITS to half of the least populated category; max: 15

if CV_SPLITS < 5:
    warnings.warn("Please provide at least 10 examples per category")

EPOCHS = 100  # (recommended), 50 will do fine too

TRAIN_FINAL = False
TRAIN_ONCE = True

OUTPUT_PROBA = False  # set to True, if you're interested in the probabilities
