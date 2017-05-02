"""
Created on Sun Apr 30 18:11:14 2017

@author: morganallen
"""

import numpy as np
from asl_data import AslDb
from asl_utils import print_sequences
from asl_utils import show_model_stats
import warnings
import timeit
from hmmlearn.hmm import GaussianHMM
from my_model_selectors import (SelectorConstant, SelectorCV, SelectorDIC, SelectorBIC)
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

#  Here we set up normalised, polar and delta features:

for i, feature in feature_index:
    df_all_mean[feature] = asl.df['speaker'].map(df_means[feature])
    df_all_std [feature] = asl.df['speaker'].map(df_std  [feature])
    asl.df[features_norm[i]] = (asl.df[feature] - df_all_mean[feature]) / df_all_std[feature]

def radius_orig(x_a, y_a):
    return [np.sqrt((x * x) + (y * y)) for (x, y) in zip(x_a, y_a)]

asl.df['polar-rr'    ] = radius_orig(asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-rtheta'] = np.arctan2 (asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-lr'    ] = radius_orig(asl.df['grnd-lx'], asl.df['grnd-ly'])
asl.df['polar-ltheta'] = np.arctan2 (asl.df['grnd-lx'], asl.df['grnd-ly'])

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
    model      = GaussianHMM(n_components = num_hidden_states, n_iter = 1000).fit(X, lengths)
    logL       = model.score(X, lengths)
    return model, logL

def train_and_show_model_stats(word, num_states, features, training_set):
    model, logL = train_a_word(word, num_states, training_set)
    show_model_stats(word, model, features)
    return model, logL

def train_with_selector(words_to_train, features, training_set, selector_class, verbose):
    sequences = training_set.get_all_sequences()
    Xlengths  = training_set.get_all_Xlengths ()
    
    for word in words_to_train:
        start = timeit.default_timer()
        model = selector_class(
            sequences,
            Xlengths,
            word,
            min_n_components = 2,
            max_n_components = 15,
            random_state     = 14,
            verbose          = verbose,
            features         = features
        ).select()
        end = timeit.default_timer()-start
        if model is not None:
            print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
        else:
            print("Training failed for {}".format(word))



print("Top rows of data:\n\n{}".format(asl.df.head()))

test_word = 'BOOK'
test_features = features_ground
test_training_set = asl.build_training(features_ground)

"""
test_word_X, test_word_lengths = test_training_set.get_word_Xlengths(test_word)
print("\n\nData on", test_word, "is:\n{}{}\n".format(test_word_X, test_word_lengths))

train_and_show_model_stats(test_word, 3, features_ground, test_training_set)

word_sequences = test_training_set.get_word_sequences(test_word)
print("\n\nWord sequences:")
print_sequences(word_sequences)

for cv_train_idx, cv_test_idx in KFold().split(word_sequences):
    print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))
"""

#  The difference between test_word_X and word_sequences is that the latter
#  simply nests each sequence within it's own array, rather than munging them
#  together and tacking on the lengths afterward.

train_with_selector([test_word], test_features, test_training_set, SelectorCV, True)



