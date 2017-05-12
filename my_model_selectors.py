
import math
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences




def show_model_stats(word, model, features):
    if model == None:
        print("No model found!")
        return
    print("Number of states trained in model for {} is {}".format(word, model.n_components))
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    for i in range(model.n_components):  # for each hidden state
        print("hidden state #{}".format(i))
        print("features   = ", features)
        print("mean       = ", model.means_[i])
        print("variance   = ", variance[i])
        print("trans %    = ", [int(p * 100) for p in model.transmat_[i]])
        print()


class ModelSelector(object):
    """
    base class for model selection (strategy design pattern)
    """
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
        warnings.filterwarnings("ignore", category=DeprecationWarning)
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
                print("  model created for {} with {} states".format(self.this_word, num_states))
            return model
        except:
            if self.verbose:
                print("  failure on {} with {} states".format(self.this_word, num_states))
            return None
    
    def select(self):
        best_model = None
        best_score = float("-inf")
        
        for p in range(self.min_n_components, self.max_n_components):
            try:
                model = self.base_model(p)
                score = self.score_model(model)
                if score > best_score:
                    best_model = model
                    best_score = score
            except Exception as e:
                print("  Selection problem:", e, "for", self.this_word)
                continue
        
        self.report_selection(best_model)
        return best_model
    
    def report_selection(self, model):
        word = self.this_word
        if self.verbose:
            show_model_stats(word, model, self.features)
        if model is not None:
            print("  Training complete for {} with {} states".format(word, model.n_components))
        else:
            print("  Training failed for {}".format(word))
    
    def score_model(self, model):
        raise NotImplementedError



class SelectorConstant(ModelSelector):
    """
    Selects the model with value self.n_constant.
    """
    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)



class SelectorBIC(ModelSelector):
    """
    Selects the model with the lowest Baysian Information Criterion(BIC) score
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    
    NOTE:  Actual model-selection is performed in the ModelSelector superclass-
    this method simply scores a pre-generated model.  (The BIC formula is also
    negated, so that the model with *highest* score can be selected.)
    """
    def score_model(self, model):
        num_D, num_F = self.X.shape
        num_S        = model.n_components
        num_params   = (num_S - 1) + (num_S * (num_S - 1)) + (2 * num_S * num_F)
        acc_bonus    = 2 * model.score(self.X, self.lengths)
        size_penalty = num_params * math.log(num_D)
        return acc_bonus - size_penalty



class SelectorDIC(ModelSelector):
    """
    Selects the best model based on Discriminative Information Criterion
    
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    
    NOTE:  Actual model-selection is performed in the ModelSelector superclass-
    this method simply scores a pre-generated model.
    """
    def score_model(self, model):
        self_score = model.score(self.X, self.lengths)
        sum_other_scores = 0
        
        for word in self.hwords.keys():
            if word == self.this_word: continue
            X, lengths = self.hwords[word]
            sum_other_scores += model.score(X, lengths)
        
        return self_score - (sum_other_scores / (len(self.hwords) - 1))


class SelectorCV(ModelSelector):
    """
    Selects the best model based on average log Likelihood of cross-validation folds.
    
    NOTE:  Actual model-selection is performed in the ModelSelector superclass-
    this method simply scores a pre-generated model.
    """
    split = None
    
    def score_model(self, model):
        #  NOTE:  KFold() chokes on small sequences, so I've hard-coded some
        #  default-splits in these cases.  Values are cached for efficiency.
        if self.split == None:
            if   len(self.sequences) == 1: self.split = [([0], [0])]
            elif len(self.sequences) == 2: self.split = [([0], [1]), ([1], [0])]
            else:                          self.split = [i for i in KFold().split(self.sequences)]
        
        sum_scores = 0
        num_scored = 0
        for train_IDs, tests_IDs in self.split:
            num_scored += 1
            train_X, train_L = combine_sequences(train_IDs, self.sequences)
            tests_X, tests_L = combine_sequences(tests_IDs, self.sequences)
            
            slice_model = self.base_model(model.n_components, train_X, train_L)
            sum_scores += slice_model.score(tests_X, tests_L)
        
        return sum_scores / num_scored








