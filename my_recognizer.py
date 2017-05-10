
import warnings
import math
from asl_data import WordsData, SinglesData
import timeit


class BasicSLM:
    
    def __init__(
        self,
        file_name: str,
        max_grams: int = 3,
        verbose    = False
    ):
        print("\nNow building SLM...")
        SLM_file = open(file_name)
        
        self.max_grams     = max_grams
        self.all_words     = set()
        self.all_freqs     = {}
        self.all_priors    = {}
        self.total_samples = 0
        
        #  Okay.  So... what do I do here?
        #  Take the total count for a given word, and divide by instances of all
        #  words.  That's the simple probability.
        
        #  If you know what the previous word was, you can get the probability of
        #  that word being followed by another, out of the set of all 2-word
        #  sequences, which also happen to start with that word.  And so on.
        
        for line in SLM_file.readlines():
            line_words = line.split()
            if verbose: print("    {}".format(line_words))
            
            for gram in range(1, max_grams + 1):
                for n in range(len(line_words) - gram):
                    sequence  = line_words[n:n + gram]
                    prior_key = str(sequence[0:-1])
                    seq_key   = str(sequence)
                    
                    if not seq_key in self.all_freqs: self.all_freqs[seq_key] = 0
                    self.all_freqs[seq_key] += 1
                    
                    if not prior_key in self.all_priors: self.all_priors[prior_key] = 0
                    self.all_priors[prior_key] += 1
                    
                    self.all_words.update(sequence)
                    self.total_samples += 1
        
        self.all_words = list(self.all_words)
        
        if verbose:
            print("\nAll words are:", self.all_words)
            
            print("\nSequences are...")
            for key in self.all_freqs.keys():
                print("  {} : {}".format(key, "|" * self.all_freqs[key]))
            
            print("\nPriors are:")
            for key in self.all_priors.keys():
                print("  {} : {}".format(key, "|" * self.all_priors[key]))
    
    
    def get_conditional_likelihood(self, sequence, smooth = 1):
        seq_key    = str(sequence)
        prior_key  = str(sequence[0:-1])
        raw_freq   = self.all_freqs[seq_key] if seq_key in self.all_freqs else 0
        priors     = self.all_priors[prior_key] if prior_key in self.all_priors else self.total_samples
        return (raw_freq + smooth) / (priors + smooth)
    
    
    def get_sample(self, recent_words: list, last_word: str):
        max_recent = self.max_grams - 1
        sample     = None
        if len(recent_words) <= max_recent: sample = recent_words.copy()
        else:                               sample = recent_words[(0 - max_recent):]
        sample.append(last_word)
        return sample


def train_all_words(
    training_set: WordsData,
    model_selector,
    word_list = None,
    verbose   = False,
    features  = []
):
    """
    Train all words given a training set and selector
    :param training: WordsData object (training set)
    :param model_selector: class (subclassed from ModelSelector)
    :return: dict of models keyed by word
    """
    sequences  = training_set.get_all_sequences()
    Xlengths   = training_set.get_all_Xlengths()
    model_dict = {}
    if word_list == None: word_list = training_set.words
    for word in word_list:
        try:
            start = timeit.default_timer()
            model = model_selector(sequences, Xlengths, word, verbose = verbose, features = features).select()
            model_dict[word] = model
            end = timeit.default_timer()-start
            if model is not None:
                if verbose: print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
            else:
                if verbose: print("Training failed for {}".format(word))
        except Exception as e:
            if verbose: print("Training failed for {}, error: {}".format(word, e))
            model_dict[word] = None
    return model_dict


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


def recognize_words(
    models:     dict,
    test_set:   SinglesData,
    lang_model: BasicSLM = None,
    word_list   = None,
    verbose     = False
):
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
                
                if lang_model != None:
                    word_sample = lang_model.get_sample(guesses, word)
                    lang_chance = lang_model.get_conditional_likelihood(word_sample)
                    score       += math.log(lang_chance)
                
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
    training_set:  WordsData,
    testing_set:   SinglesData,
    model_selector ,
    lang_model:    BasicSLM = None,
    train_words    = None,
    test_words     = None,
    verbose        = False,
    features       = []
):
    print("\n\nPERFORMING RECOGNITION PASS...")
    
    models_dict = train_all_words(training_set, model_selector, train_words, verbose, features)
    
    if test_words == None: test_words = testing_set.wordlist
    probabilities, guesses = recognize_words(models_dict, testing_set, lang_model, test_words, verbose)
    
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
    print("\n  FEATURES ARE:", features)
    print("  LANG MODEL IS: ", lang_model)
    print("  SELECTOR IS: ", model_selector)
    print("  ACCURACY: {}%".format(accuracy))


