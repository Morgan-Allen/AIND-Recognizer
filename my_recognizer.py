
import warnings
import math
import json
from asl_data import SinglesData


class BasicSLM:
    def __init__(
        self,
        file_name: str,
        max_grams: int = 3,
        verbose    = False
    ):
        """
        Compiles frequency tables for all n-gram sequences and their priors up
        to a given gram limit.
        """
        if verbose: print("\nNow building SLM...")
        
        self.max_grams     = max_grams
        self.all_words     = set()
        self.all_freqs     = {}
        self.all_priors    = {}
        self.total_samples = 0
        
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
    
    
    def get_sample(self, guesses: list, index: int, guess_word: str):
        """
        Samples up to max_grams - 1 words from the list of previous guesses,
        then attaches the potential guess to that list.
        """
        max_recent = self.max_grams - 1
        sample     = None
        if index + 1 <= max_recent: sample = guesses[0:index]
        else:                       sample = guesses[index - max_recent:index]
        sample.append(guess_word)
        return sample
    
    
    def get_conditional_prob(self, sample, smooth = 1):
        """
        Returns the conditional probability of the last word in the sample of
        recent words.  An optional smoothing parameter is used to handle
        unknown words.
        """
        seq_key   = str(sample)
        prior_key = str(sample[0:-1])
        raw_freq  = self.all_freqs[seq_key] if seq_key in self.all_freqs else 0
        priors    = self.all_priors[prior_key] if prior_key in self.all_priors else self.total_samples
        return (raw_freq + smooth) / (priors + smooth)
    
    
    def get_score(self, guesses: list, probabilities: list, index: int, guess: str):
        sample = self.get_sample(guesses, index, guess)
        return math.log(self.get_conditional_prob(sample))




class FuzzySLM:
    
    
    class Word:
        def __init__(self):
            self.label     = None
            self.word_type = None
            self.after     = {}
            self.before    = {}
        
        def is_type(self):
            return self.label == self.word_type
        
        def normalise_table(self, link_table):
            sum_words = 0
            sum_types = 0
            
            for key in link_table:
                if key.is_type(): sum_types += link_table[key]
                else:             sum_words += link_table[key]
            
            for key in link_table:
                if key.is_type(): link_table[key] /= sum_types
                else:             link_table[key] /= sum_words
        
        def normalise(self):
            self.normalise_table(self.after )
            self.normalise_table(self.before)
        
        def after_weight(self, after_word):
            if not after_word in self.after: return 0
            return self.after[after_word]
        
        def before_weight(self, before_word):
            if not before_word in self.before: return 0
            return self.before[before_word]
        
        def __repr__(self):
            return self.label
        
        def print(self):
            print("\n  Label:    {}".format(self.label))
            print("  Category: {}".format(self.word_type))
            
            afters = list(self.after.keys())
            def sort_weights(key): return 0 - self.after[key]
            afters = sorted(afters, key = sort_weights)
            print("  These come after:")
            for key in afters:
                print("    {:<16.16}: {:<10.10}".format(str(key), str(self.after[key])))
    
    
    def inc_weight(self, word_label, prior_label, weight):
        word  = self.words[word_label ]
        prior = self.words[prior_label]
        if not word  in prior.after : prior.after [word ] = 0
        if not prior in word .before: word .before[prior] = 0
        prior.after[word ] += weight
        word.before[prior] += weight
    
    
    def get_sample(self, words, index, num_words):
        sample = None
        if index + 1 <= num_words: sample = words[0:index]
        else:                      sample = words[index - num_words:index]
        return sample
    
    
    def __init__(
        self,
        grammar_filename: str,
        corpus_filename:  str,
        max_grams:        int = 3,
        verbose           = False
    ):
        grammar_file = open(grammar_filename, 'r')
        grammar      = json.load(grammar_file)
        
        self.max_grams = max_grams
        self.word_list = []
        self.type_list = []
        self.words     = {}
        
        for type_label in grammar.keys():
            self.words[type_label] = word_type = self.Word()
            word_type.label        = type_label
            word_type.word_type    = type_label
            self.type_list.append(type_label)
            
            for label in grammar[type_label]:
                self.words[label] = word = self.Word()
                word.label        = label
                word.word_type    = type_label
                self.word_list.append(label)
        
        corpus_file = open(corpus_filename, 'r')
        for line in corpus_file.readlines():
            line_words = line.split()
            
            for i in range(len(line_words)):
                last_word = line_words[i]
                last_type = self.words[last_word].word_type
                
                sample = self.get_sample(line_words, i, max_grams)
                weight = 1.
                for word in sample:
                    word_type = self.words[word].word_type
                    self.inc_weight(last_word, word     , weight / 1)
                    self.inc_weight(last_type, word_type, weight / 2)
                    self.inc_weight(last_type, word     , weight / 2)
                    self.inc_weight(last_word, word_type, weight / 4)
                    weight /= 2
        
        for word in self.words.values():
            word.normalise()
        
        if verbose:
            print("\nGenerated Fuzzy SLM:")
            for key in self.word_list:
                word = self.words[key]
                word.print()
            print("\nGrammar Types:")
            for key in self.type_list:
                word = self.words[key]
                word.print()
    
    
    def get_score(self, guesses: list, probabilities: list, index: int, guess: str):
        if not guess in self.words or index == 0: return 0
        
        sample     = self.get_sample(guesses, index, self.max_grams)
        guess      = self.words[guess]
        guess_type = self.words[guess.word_type]
        score      = 0.0
        weight     = 2.0
        
        for label in sample[::-1]:
            weight /= 2
            if not label in self.words: continue
            
            word      = self.words[label]
            word_type = self.words[word.word_type]
            score     += word     .after_weight(guess     ) * weight / 1
            score     += word     .after_weight(guess_type) * weight / 3.33
            score     += word_type.after_weight(guess     ) * weight / 3.33
            score     += word_type.after_weight(guess_type) * weight / 3.33
        
        if score == 0: return 0
        score /= 2 * len(sample)
        return math.log(score)


def normalise_probs(probs):
    """
    Normalises all probabilities for a given word to fit within the range of 0
    to 1000, and returns the normalised table.
    """
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



def recognize(models: dict, test_set: SinglesData):
    """
      Recognize test word sequences from word models set
      :param models: dict of trained models
        {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': etc...}
      :param test_set: SinglesData object
     
      :return: (list, list)  as probabilities, guesses
        both lists are ordered by the test set word_id
        probabilities is a list of dictionaries where each key a word and value is Log likelihood
          [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
           {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },]
        guesses is a list of the best guess words ordered by the test set word_id
          ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
    """
    return recognize_words(models, test_set, test_set.wordlist)


def recognize_words(
    models:     dict,
    test_set:   SinglesData,
    word_list   = None,
    verbose     = False
):
    """
    Similar to recognize, but with optional parameters to limit recognition to
    a particular subset of words or add verbosity.
    """
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


def scale_and_combine(
    all_guesses  ,
    all_probs    ,
    SLM          ,
    SLM_weight   = 1.0
):
    """
    Combines probabilities from both the recognizer and SLM by normalising
    each, averaging with the supplied weight, normalising the result, and
    returning a tuple with the new probabilities and new guesses.
    """
    all_guesses = all_guesses.copy()
    all_new_probs, all_new_guesses = [], []
    
    for index in range(len(all_guesses)):
        probs      = all_probs[index]
        SLM_probs  = {}
        new_probs  = {}
        best_score = float("-inf")
        best_guess = None
        
        for guess in probs.keys():
            SLM_probs[guess] = SLM.get_score(all_guesses, all_probs, index, guess)
        
        for guess in probs.keys():
            new_score = probs[guess] + (SLM_probs[guess] * SLM_weight)
            new_probs[guess] = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_guess = guess
        
        all_new_probs  .append(new_probs )
        all_new_guesses.append(best_guess)
        
        all_guesses[index] = best_guess
    
    return all_new_probs, all_new_guesses



def report_recognizer_results(
    test_words    ,
    probabilities ,
    guesses       ,
    model_selector,
    lang_model    ,
    features
):
    """
    Reports on recognizer accuracy with various parameters
    """
    word_ID  = 0
    num_hits = 0
    num_miss = 0
    
    print("\n\nREPORTING RECOGNIZER RESULTS.")
    for word in test_words:
        guess = guesses[word_ID]
        if guess == word: num_hits += 1
        else:             num_miss += 1
        word_ID += 1
    
    accuracy = (100 * num_hits) / (num_hits + num_miss)
    print("\n  FEATURES ARE:", features)
    print("  LANGUAGE MODEL IS: ", lang_model)
    print("  SELECTOR IS: ", model_selector)
    print("  ACCURACY: {}%".format(accuracy))
    
    return accuracy


