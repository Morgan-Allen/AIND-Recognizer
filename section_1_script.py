#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 18:11:14 2017

@author: morganallen
"""


import numpy as np
import pandas as pd
from asl_data import AslDb


# initializes the database anddisplays the first five rows of the asl database,
# indexed by video and frame
asl = AslDb()
asl.df.head() 

# look at the data available for an individual frame
asl.df.ix[98,1]

# the new feature 'grnd-ry' is now in the frames dictionary
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x' ] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y' ] - asl.df['nose-y']
asl.df.head()


from asl_utils import test_features_tryit
# test the code
test_features_tryit(asl)


# collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
# show a single set of features for a given (video, frame) tuple
[asl.df.ix[98,1][v] for v in features_ground]

training = asl.build_training(features_ground)
print("Training words: {}".format(training.words))

training.get_word_Xlengths('CHOCOLATE')


df_means = asl.df.groupby('speaker').mean()
df_means

asl.df.head()

from asl_utils import test_std_tryit
df_std = asl.df.groupby('speaker').std()

# test the code
test_std_tryit(df_std)



# TODO add features for normalized by speaker values of left, right, x, y
# Name these 'norm-rx', 'norm-ry', 'norm-lx', and 'norm-ly'
# using Z-score scaling (X-Xmean)/Xstd

features_core = ['right-x', 'right-y', 'left-x' , 'left-y' ]
features_norm = ['norm-rx', 'norm-ry', 'norm-lx', 'norm-ly']
feature_index = [(i, features_core[i]) for i in range(len(features_core))]
df_all_mean   = {}
df_all_std    = {}

for i, feature in feature_index:
    df_all_mean[feature] = asl.df['speaker'].map(df_means[feature])
    df_all_std [feature] = asl.df['speaker'].map(df_std  [feature])
    asl.df[features_norm[i]] = (asl.df[feature] - df_all_mean[feature]) / df_all_std[feature]



# TODO add features for polar coordinate values where the nose is the origin
# Name these 'polar-rr', 'polar-rtheta', 'polar-lr', and 'polar-ltheta'
# Note that 'polar-rr' and 'polar-rtheta' refer to the radius and angle

features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

def radius_orig(x_a, y_a):
    return [np.sqrt((x * x) + (y * y)) for (x, y) in zip(x_a, y_a)]

asl.df['polar-rr'    ] = radius_orig(asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-rtheta'] = np.arctan2 (asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-lr'    ] = radius_orig(asl.df['grnd-lx'], asl.df['grnd-ly'])
asl.df['polar-ltheta'] = np.arctan2 (asl.df['grnd-lx'], asl.df['grnd-ly'])



# TODO add features for left, right, x, y differences by one time step, i.e. the "delta" values discussed in the lecture
# Name these 'delta-rx', 'delta-ry', 'delta-lx', and 'delta-ly'

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

for i, feature in feature_index:
    data_diff = np.lib.pad(np.diff(asl.df[feature]), (1,0), 'constant', constant_values=(0, 0))
    asl.df[features_delta[i]] = data_diff


# TODO add features of your own design, which may be a combination of the above or something else
# Name these whatever you would like

# TODO define a list named 'features_custom' for building the training set
features_custom = []


import unittest
# import numpy as np

class TestFeatures(unittest.TestCase):

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






