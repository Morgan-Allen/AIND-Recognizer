"""
Created on Sun Apr 30 18:11:14 2017

@author: morganallen
"""

import warnings
import numpy as np
from asl_data import AslDb
from my_model_selectors import (SelectorCV, SelectorDIC, SelectorBIC)
from my_recognizer import perform_recognizer_pass


# initializes the database

asl = AslDb()

features_core   = ['right-x', 'right-y', 'left-x' , 'left-y' ]
features_ground = ['grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']
features_norm   = ['norm-rx', 'norm-ry', 'norm-lx', 'norm-ly']
features_polar  = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
features_delta  = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']
features_custom = ['ngrnd-rx', 'ngrnd-ry', 'ngrnd-lx', 'ngrnd-ly']

asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x' ] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y' ] - asl.df['nose-y']

df_means     = asl.df.groupby('speaker').mean()
df_std       = asl.df.groupby('speaker').std ()
core_index   = [(i, features_core  [i]) for i in range(len(features_core  ))]
ground_index = [(i, features_ground[i]) for i in range(len(features_ground))]
df_all_mean  = {}
df_all_std   = {}

#  Here we set up normalised, polar, delta and normalised-ground features:

def radius_orig(x_a, y_a):
    return [np.sqrt((x * x) + (y * y)) for (x, y) in zip(x_a, y_a)]

asl.df['polar-rr'    ] = radius_orig(asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-rtheta'] = np.arctan2 (asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-lr'    ] = radius_orig(asl.df['grnd-lx'], asl.df['grnd-ly'])
asl.df['polar-ltheta'] = np.arctan2 (asl.df['grnd-lx'], asl.df['grnd-ly'])

for i, feature in core_index:
    feature_mean = asl.df['speaker'].map(df_means[feature])
    feature_std  = asl.df['speaker'].map(df_std  [feature])
    feature_diff = np.lib.pad(np.diff(asl.df[feature]), (1,0), 'constant', constant_values=(0, 0))
    asl.df[features_norm[i]] = (asl.df[feature] - feature_mean) / feature_std
    asl.df[features_delta[i]] = feature_diff

for i, feature in ground_index:
    feature_mean = asl.df['speaker'].map(df_means[feature])
    feature_std  = asl.df['speaker'].map(df_std  [feature])
    asl.df[features_custom[i]] = (asl.df[feature] - feature_mean) / feature_std



print("Top rows of data:\n\n{}".format(asl.df.head()))

all_selectors = [SelectorBIC, SelectorDIC, SelectorCV]
feature_sets  = [features_ground, features_polar, features_custom]


warnings.filterwarnings("ignore")

for feature_set in feature_sets:
    for selector in all_selectors:
        training_set   = asl.build_training(feature_set)
        testing_set    = asl.build_test    (feature_set)
        train_words    = training_set.words
        #train_words    = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
        test_words     = testing_set.wordlist
        #test_words     = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
        perform_recognizer_pass(training_set, testing_set, selector, train_words, test_words, features=feature_set)



