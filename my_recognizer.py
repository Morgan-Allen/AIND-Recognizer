import warnings
from asl_data import WordsData, SinglesData
from asl_utils import train_all_words, report_recognize_results


"""
Recognize test word sequences from word models set

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
def recognize(models: dict, test_set: SinglesData):
    return recognize_words(models, test_set, test_set.wordlist)


def recognize_words(models: dict, test_set: SinglesData, word_list, verbose = False):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses       = []
    
    for word_id in word_list:
        best_score = float("-inf")
        best_guess = None
        
        if verbose: print("Attempting to recognise", word_id)
        try:
            index = test_set.wordlist.index(word_id)
            word_X, word_L = test_set.get_item_Xlengths(index)
            for word in models.keys():
                model = models[word]
                if model == None: continue
                score = model.score(word_X, word_L)
                if verbose: print("  Score for", word, "is", score)
                
                if score > best_score:
                    best_score = score
                    best_guess = word
            
        except Exception as e:
            print("Recognition error:", e)
            pass
        
        if verbose: print("  Best guess:", best_guess, "Score:", best_score)
        probabilities.append(best_score)
        guesses      .append(best_guess)
    
    return probabilities, guesses


def perform_recognizer_pass(
    training_set: WordsData,
    testing_set: SinglesData,
    model_selector,
    train_words = None,
    test_words = None,
    verbose = False,
    features = []
):
    models_dict = train_all_words(training_set, model_selector, train_words, verbose, features)
    if test_words == None: test_words = testing_set.wordlist
    
    probabilities, guesses = recognize_words(models_dict, testing_set, test_words, verbose)
    report_recognize_results(probabilities, guesses, test_words)




