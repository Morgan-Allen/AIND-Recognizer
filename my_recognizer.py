
import warnings
from asl_data import WordsData, SinglesData
from asl_utils import train_all_words


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
        all_probs  = {}
        
        if verbose: print("  Attempting to recognise", word_id)
        try:
            index = test_set.wordlist.index(word_id)
            word_X, word_L = test_set.get_item_Xlengths(index)
            for word in models.keys():
                score = float("-inf")
                try:
                    model = models[word]
                    score = model.score(word_X, word_L)
                except Exception as e:
                    if verbose: print("  Recognition error:", e, "for", word, "in", word_id)
                
                all_probs[word] = score
                if verbose: print("  Score for", word, "is", score)
                if score > best_score:
                    best_score = score
                    best_guess = word
                
        except Exception as e:
            if verbose: print("  Recognition error:", e, "for", word_id)
            pass
        
        if verbose: print("  Best guess:", best_guess, "Score:", best_score)
        probabilities.append(all_probs )
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
    print("\n\nPERFORMING RECOGNITION PASS...")
    
    models_dict = train_all_words(training_set, model_selector, train_words, verbose, features)
    if test_words == None: test_words = testing_set.wordlist
    probabilities, guesses = recognize_words(models_dict, testing_set, test_words, verbose)
    
    word_ID  = 0
    num_hits = 0
    num_miss = 0
    
    for word in test_words:
        guess = guesses      [word_ID]
        prob  = probabilities[word_ID][guess]
        print("  Guess for", word, "is", guess, "log. prob:", prob)
        
        if guess == word: num_hits += 1
        else:             num_miss += 1
        word_ID += 1
    
    accuracy = (100 * num_hits) / (num_hits + num_miss)
    print("\n  FEATURES ARE:", features      )
    print("  SELECTOR IS: ", model_selector)
    print("  ACCURACY: {}%".format(accuracy))





