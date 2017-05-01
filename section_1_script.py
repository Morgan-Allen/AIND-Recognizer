"""
Created on Sun Apr 30 18:11:14 2017

@author: morganallen
"""

import numpy as np
import math
import pandas as pd
from asl_data import AslDb
from asl_utils import test_features_tryit
from asl_utils import test_std_tryit
import unittest
import warnings
from hmmlearn.hmm import GaussianHMM
from my_model_selectors import SelectorConstant
from sklearn.model_selection import KFold



# initializes the database

asl = AslDb()

features_core   = ['right-x', 'right-y', 'left-x' , 'left-y' ]
features_ground = ['grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']
features_norm   = ['norm-rx', 'norm-ry', 'norm-lx', 'norm-ly']
features_polar  = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
features_delta  = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']
features_custom = []

asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x' ] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y' ] - asl.df['nose-y']

df_means = asl.df.groupby('speaker').mean()
df_std   = asl.df.groupby('speaker').std ()
feature_index = [(i, features_core[i]) for i in range(len(features_core))]
df_all_mean   = {}
df_all_std    = {}


#  Here we set up normalised features:

for i, feature in feature_index:
    df_all_mean[feature] = asl.df['speaker'].map(df_means[feature])
    df_all_std [feature] = asl.df['speaker'].map(df_std  [feature])
    asl.df[features_norm[i]] = (asl.df[feature] - df_all_mean[feature]) / df_all_std[feature]

#  Here we set up polar coordinates:

def radius_orig(x_a, y_a):
    return [np.sqrt((x * x) + (y * y)) for (x, y) in zip(x_a, y_a)]

asl.df['polar-rr'    ] = radius_orig(asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-rtheta'] = np.arctan2 (asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-lr'    ] = radius_orig(asl.df['grnd-lx'], asl.df['grnd-ly'])
asl.df['polar-ltheta'] = np.arctan2 (asl.df['grnd-lx'], asl.df['grnd-ly'])

#  Here we set up delta coordinates:

for i, feature in feature_index:
    data_diff = np.lib.pad(np.diff(asl.df[feature]), (1,0), 'constant', constant_values=(0, 0))
    asl.df[features_delta[i]] = data_diff

# TODO add features of your own design, which may be a combination of the above or something else
# Name these whatever you would like



#  NOTE:  The model will compute covariance between each of it's internal component-dimensions
#  and all the features (though often you'll only be looking for a match between the [i]th
#  feature and the i[th] dimension, as by default.)  The covariance is stored in matrix form,
#  With i-to-i covariance stored along the diagonal.

def train_a_word(word, num_hidden_states, training_set):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    X, lengths = training_set.get_word_Xlengths(word)
    model      = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
    logL       = model.score(X, lengths)
    return model, logL

def show_model_stats(word, model, features):
    print("Number of states trained in model for {} is {}".format(word, model.n_components))
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    for i in range(model.n_components):  # for each hidden state
        print("hidden state #{}".format(i))
        print("features   = ", features)
        print("mean       = ", model.means_[i])
        print("variance   = ", variance[i])
        print("trans %    = ", [int(p * 100) for p in model.transmat_[i]])
        print()

def train_and_show_model_stats(word, num_states, features, training_set):
    model, logL = train_a_word(word, num_states, training_set)
    show_model_stats(word, model, features)
    return model


print("Top rows of data: ", asl.df.head())
ground_training_set = asl.build_training(features_ground)

train_and_show_model_stats('BOOK', 3, features_ground, ground_training_set)




"""

demoword = 'BOOK'
model, logL = train_a_word(demoword, 3, features_ground)
print("Number of states trained in model for {} is {}".format(demoword, model.n_components))
print("logL = {}".format(logL))

show_model_stats(demoword, model, features_ground)

my_testword = 'CHOCOLATE'
model, logL = train_a_word(my_testword, 3, features_ground) # Experiment here with different parameters
show_model_stats(my_testword, model, features_ground)
print("logL = {}".format(logL))




training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
word = 'VEGETABLE' # Experiment here with different words
model = SelectorConstant(training.get_all_sequences(), training.get_all_Xlengths(), word, n_constant=3).select()
print("Number of states trained in model for {} is {}".format(word, model.n_components))



# Experiment here with different feature sets
# Experiment here with different words
# view indices of the folds

training = asl.build_training(features_ground) 
word = 'VEGETABLE' 
word_sequences = training.get_word_sequences(word)
split_method = KFold()
for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
    print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))
"""



"""
words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
import timeit


# TODO: Implement SelectorCV in my_model_selector.py
from my_model_selectors import SelectorCV

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorCV(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))



# TODO: Implement SelectorBIC in module my_model_selectors.py
from my_model_selectors import SelectorBIC

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorBIC(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))



# TODO: Implement SelectorDIC in module my_model_selectors.py
from my_model_selectors import SelectorDIC

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorDIC(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))



from asl_test_model_selectors import TestSelectors
suite = unittest.TestLoader().loadTestsFromModule(TestSelectors())
unittest.TextTestRunner().run(suite)
"""


"""
# print("Training words: {}".format(training.words))
# training.get_word_Xlengths('CHOCOLATE')

class TestFeatures(unittest.TestCase):
    
    def test_features_basic(self):
        test_features_tryit(asl)
        test_std_tryit(df_std)
        self.assertEqual(True, True)
    
    def test_features_ground(self):
        sample = (asl.df.ix[98, 1][features_ground]).tolist()
        self.assertEqual(sample, [9, 113, -12, 119])

    def test_features_norm(self):
        sample = (asl.df.ix[98, 1][features_norm]).tolist()
        np.testing.assert_almost_equal(sample, [ 1.153,  1.663, -0.891,  0.742], 3)

    def test_features_polar(self):
        sample = (asl.df.ix[98,1][features_polar]).tolist()
        np.testing.assert_almost_equal(sample, [113.3578, 0.0794, 119.603, -0.1005], 3)

    def test_features_delta(self):
        sample = (asl.df.ix[98, 0][features_delta]).tolist()
        self.assertEqual(sample, [0, 0, 0, 0])
        sample = (asl.df.ix[98, 18][features_delta]).tolist()
        self.assertTrue(sample in [[-16, -5, -2, 4], [-14, -9, 0, 0]], "Sample value found was {}".format(sample))
                         
suite = unittest.TestLoader().loadTestsFromModule(TestFeatures())
unittest.TextTestRunner().run(suite)

"""


