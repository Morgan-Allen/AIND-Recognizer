import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses       = []
    
    for word_id in test_set.wordlist:
        word_X, word_L = test_set.get_item_Xlengths(word_id)
        best_score = float("-inf")
        best_guess = None
        
        for word in models.keys():
            model = models[word]
            score = model.score(word_X, word_L)
            if score > best_score:
                best_score = score
                best_guess = word
        
        probabilities.append(best_score)
        guesses       .append(best_guess)
    
    return probabilities, guesses
