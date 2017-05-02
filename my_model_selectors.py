
import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
from asl_utils import print_sequences
from asl_utils import print_X_and_L
from asl_utils import show_model_stats



class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''
    def __init__(
        self,
        all_word_sequences: dict,
        all_word_Xlengths:  dict,
        this_word:          str,
        n_constant          = 3,
        min_n_components    = 2,
        max_n_components    = 10,
        random_state        = 14,
        verbose             = False,
        features            = []
    ):
        self.words            = all_word_sequences
        self.hwords           = all_word_Xlengths
        self.sequences        = all_word_sequences[this_word]
        self.X, self.lengths  = all_word_Xlengths [this_word]
        self.this_word        = this_word
        self.n_constant       = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state     = random_state
        self.verbose          = verbose
        self.features         = features
    
    def base_model(self, num_states, sequences = None, lengths = None):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            model = GaussianHMM(
                n_components    = num_states,
                covariance_type = "diag",
                n_iter          = 1000,
                random_state    = self.random_state,
                verbose         = False
            )
            if sequences == None: sequences = self.X
            if lengths != None: model.fit(sequences, lengths)
            else:               model.fit(sequences)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None
    
    def select(self):
        picked = None
        best_score = float("inf")
        
        for p in range(self.min_n_components, self.max_n_components):
            model = self.base_model(p)
            score = self.score_model(model)
            if score < best_score:
                picked = model
                best_score = score
        
        if self.verbose:
            show_model_stats(self.this_word, picked, self.features)
        
        return picked
    
    def score_model(self, model):
        raise NotImplementedError


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """
    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def score_model(self, model):
        num_S, num_F = model.n_components, len(model.means_[0])
        num_params = (num_S - 1) + (num_S * (num_S - 1)) + (2 * num_S * num_F)
        acc_bonus   = -2 * model.score(self.X, self.lengths)
        len_penalty = num_params * math.log(len(self.lengths))
        return acc_bonus + len_penalty


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion
    
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def score_model(self, model):
        self_score = model.score(self.X, self.lengths)
        sum_other_scores = 0
        
        for word in self.hwords.keys():
            if word == self.this_word: continue
            X, lengths = self.hwords[word]
            sum_other_scores += model.score(X, lengths)
        
        return (sum_other_scores / (len(self.hwords) - 1)) - self_score


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds.
    '''
    
    def select(self):
        split      = KFold().split(self.sequences)
        picked     = None
        best_score = float("-inf")
        
        for train_IDs, tests_IDs in split:
            train_X, train_L = combine_sequences(train_IDs, self.sequences)
            tests_X, tests_L = combine_sequences(tests_IDs, self.sequences)
            
            if self.verbose:
                print_X_and_L(train_X, train_L, "Train on:")
                print_X_and_L(tests_X, tests_L, "Tests on:")
            
            for p in range(self.min_n_components, self.max_n_components):
                model = self.base_model(p, train_X, train_L)
                score = model.score(tests_X, tests_L)
                
                if score > best_score:
                    picked     = model
                    best_score = score
        
        if self.verbose:
            show_model_stats(self.this_word, picked, self.features)
        
        return picked








