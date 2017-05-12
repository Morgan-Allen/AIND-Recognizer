
import warnings
import math
from asl_data import SinglesData


class BasicSLM:
    
    def __init__(
        self,
        file_name: str,
        max_grams: int = 3,
        verbose    = False
    ):
        print("\nNow building SLM...")
        
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
        
        SLM_file = open(file_name)
        for line in SLM_file.readlines():
            line_words = line.split()
            if verbose: print("    {}".format(line_words))
            
            for gram in range(1, max_grams + 1):
                for n in range(len(line_words) + 1 - gram):
                    sequence  = line_words[n:n + gram]
                    prior_key = str(sequence[0:-1])
                    seq_key   = str(sequence)
                    
                    if not seq_key in self.all_freqs: self.all_freqs[seq_key] = 0
                    self.all_freqs[seq_key] += 1
                    
                    if not prior_key in self.all_priors: self.all_priors[prior_key] = 0
                    self.all_priors[prior_key] += 1
                    
                    self.all_words.update(sequence)
                    self.total_samples += 1
                    if gram > 1 and verbose: print("      {}".format(sequence))
        
        self.all_words = list(self.all_words)
        
        if verbose:
            print("\nAll words are:", self.all_words)
            
            print("\nSequences are...")
            for key in self.all_freqs.keys():
                print("  {} : {}".format(key, "|" * self.all_freqs[key]))
            
            print("\nPriors are:")
            for key in self.all_priors.keys():
                print("  {} : {}".format(key, "|" * self.all_priors[key]))
    
    
    def get_sample(self, test_words: list, index: int, guess_word: str):
        max_recent = self.max_grams - 1
        sample     = None
        if index + 1 <= max_recent: sample = test_words[0:index]
        else:                       sample = test_words[index - max_recent:index]
        sample.append(guess_word)
        return sample
    
    
    def get_conditional_prob(self, sample, smooth = 1):
        seq_key   = str(sample)
        prior_key = str(sample[0:-1])
        raw_freq  = self.all_freqs[seq_key] if seq_key in self.all_freqs else 0
        priors    = self.all_priors[prior_key] if prior_key in self.all_priors else self.total_samples
        return (raw_freq + smooth) / (priors + smooth)
    
    
    def get_max_prob(self, sample):
        #"""
        return self.get_conditional_prob(sample)
        #"""
        
        """
        seq_key = str(sample)
        if seq_key in self.all_freqs:
            return self.get_conditional_prob(sample)
        """
        
        """
        #  TODO:  This isn't actually improving the estimate in any way.
        max_prob = 0
        divisor  = 1.
        gram     = min(len(sample), self.max_grams)
        while gram > 0:
            sub_sample = sample[0 - gram:]
            prob       = self.get_conditional_prob(sub_sample)
            max_prob   = max_prob + (prob / divisor)
            gram       -= 1
            divisor    *= 100
            break
        return max_prob
        """


def normalise_probs(probs):
    max_prob = float("-inf")
    min_prob = float("inf")
    new_probs = {}
    
    for key in probs.keys():
        prob = probs[key]
        if prob == float("inf") or prob == float("-inf"): continue
        max_prob = max(max_prob, prob)
        min_prob = min(min_prob, prob)
    
    prob_range = max_prob - min_prob
    if prob_range == 0: prob_range = 1
    
    for key in probs.keys():
        prob = probs[key]
        if prob < min_prob: prob = min_prob
        if prob > max_prob: prob = max_prob
        new_probs[key] = (prob - min_prob) * 1000. / prob_range
    
    return new_probs



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


def get_SLM_probs(all_words, all_probs, SLM):
    all_SLM_probs = []
    
    for index in range(len(all_words)):
        probs = all_probs[index]
        SLM_probs = {}
        for guess in probs.keys():
            sample   = SLM.get_sample(all_words, index, guess)
            SLM_prob = SLM.get_max_prob(sample)
            SLM_probs[guess] = math.log(SLM_prob)
        
        all_SLM_probs.append(SLM_probs)
    
    return all_SLM_probs


def normalise_and_combine(
    all_words    ,
    all_probs    ,
    all_SLM_probs,
    all_guesses  ,
    SLM_weight   = 1.0
):
    all_new_probs, all_new_guesses = [], []
    
    for index in range(len(all_words)):
        probs      = normalise_probs(all_probs    [index])
        SLM_probs  = normalise_probs(all_SLM_probs[index])
        new_probs  = {}
        best_score = float("-inf")
        best_guess = None
        
        for guess in probs.keys():
            new_score = probs[guess] + (SLM_probs[guess] * SLM_weight)
            new_probs[guess] = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_guess = guess
        
        new_probs = normalise_probs(new_probs)
        all_new_probs  .append(new_probs )
        all_new_guesses.append(best_guess)
    
    return all_new_probs, all_new_guesses


def report_recognizer_results(
    test_words    ,
    probabilities ,
    guesses       ,
    model_selector,
    lang_model    ,
    features
):
    word_ID  = 0
    num_hits = 0
    num_miss = 0
    
    print("\n\nREPORTING RECOGNIZER RESULTS.")
    for word in test_words:
        guess = guesses      [word_ID]
        prob  = probabilities[word_ID][guess]
        print("  Guess for", word, "is", guess, "log. prob:", prob)
        
        if guess == word: num_hits += 1
        else:             num_miss += 1
        word_ID += 1
    
    accuracy = (100 * num_hits) / (num_hits + num_miss)
    print("\n  FEATURES ARE:", features)
    print("  LANGUAGE MODEL IS: ", lang_model)
    print("  SELECTOR IS: ", model_selector)
    print("  ACCURACY: {}%".format(accuracy))
    
    return accuracy


