"""
Created on Sun Apr 30 18:11:14 2017

@author: morganallen
"""

import numpy as np
from asl_data import AslDb
from my_model_selectors import (SelectorConstant, SelectorCV, SelectorDIC, SelectorBIC)
from my_recognizer import perform_recognizer_pass


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



print("Top rows of data:\n\n{}".format(asl.df.head()))

train_features = features_ground
training_set   = asl.build_training(train_features)
testing_set    = asl.build_test    (train_features)
#train_words    = training_set.words
train_words    = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
#test_words     = testing_set.wordlist
test_words     = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']

#perform_recognizer_pass(training_set, testing_set, SelectorConstant, train_words, test_words)
perform_recognizer_pass(training_set, testing_set, SelectorBIC     , train_words, test_words)
#perform_recognizer_pass(training_set, testing_set, SelectorDIC     , train_words, test_words)
#perform_recognizer_pass(training_set, testing_set, SelectorCV      , train_words, test_words)






