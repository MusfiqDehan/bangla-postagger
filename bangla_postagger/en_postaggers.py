"""
This module produce differerent English PoS Taggers for a given sentence.
"""

from textblob import TextBlob
from flair.models import SequenceTagger
from flair.data import Sentence
import en_core_web_sm
import spacy.cli
import spacy
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

spacy.cli.download("en_core_web_sm")


# =================================
# converting words into pos tags
# =================================
def words_to_tags(sentence):
    sentence = nltk.tokenize.word_tokenize(sentence)
    tagged_sentence = nltk.pos_tag(sentence)
    tags = [tag for word, tag in tagged_sentence]
    tags_str = ' '.join([str(tag) for tag in tags])
    return tags_str


def get_correct_spelling(sentence: str) -> str:
    '''
    Get correct spelling of a sentence.\n
    At first install dependencies \n
    `!pip install -U textblob`
    '''
    correct_spelling = TextBlob(sentence).correct()
    return correct_spelling


# ========================================
# Get Words Tags Dictionary
# ========================================
def get_nltk_postag_dict(target=""):
    ''' 
    Get nltk pos tags 
    '''
    target_tokenized = nltk.tokenize.word_tokenize(target)
    nltk_postag_dict = dict((key, value)
                            for key, value in nltk.pos_tag(target_tokenized))
    return nltk_postag_dict


def get_spacy_postag_dict(target=""):
    ''' 
    Get spacy pos tags 
    '''
    nlp = en_core_web_sm.load()
    target_tokenized = nlp(target)
    spacy_postag_dict = dict((token.text, token.tag_)
                             for token in target_tokenized)
    return spacy_postag_dict


def get_flair_postag_dict(target=""):
    ''' 
    Get flair pos tags 
    '''
    tagger = SequenceTagger.load("pos")
    target_tokenized = Sentence(target)
    tagger.predict(target_tokenized)
    flair_postag_dict = dict((token.text, token.tag)
                             for token in target_tokenized)
    return flair_postag_dict
