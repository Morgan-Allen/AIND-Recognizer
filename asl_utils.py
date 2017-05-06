from asl_data import SinglesData, WordsData
import numpy as np
from IPython.core.display import display, HTML
from hmmlearn.hmm import GaussianHMM
import warnings
import timeit


RAW_FEATURES = ['left-x', 'left-y', 'right-x', 'right-y']
GROUND_FEATURES = ['grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']


def show_errors(guesses: list, test_set: SinglesData):
    """ Print WER and sentence differences in tabular form

    :param guesses: list of test item answers, ordered
    :param test_set: SinglesData object
    :return:
        nothing returned, prints error report

    WER = (S+I+D)/N  but we have no insertions or deletions for isolated words so WER = S/N
    """
    S = 0
    N = len(test_set.wordlist)
    num_test_words = len(test_set.wordlist)
    if len(guesses) != num_test_words:
        print("Size of guesses must equal number of test words ({})!".format(num_test_words))
    for word_id in range(num_test_words):
        if guesses[word_id] != test_set.wordlist[word_id]:
            S += 1
    
    print("\n**** WER = {}".format(float(S) / float(N)))
    print("Total correct: {} out of {}".format(N - S, N))
    print('Video  Recognized                                                    Correct')
    print('=====================================================================================================')
    for video_num in test_set.sentences_index:
        correct_sentence = [test_set.wordlist[i] for i in test_set.sentences_index[video_num]]
        recognized_sentence = [guesses[i] for i in test_set.sentences_index[video_num]]
        for i in range(len(recognized_sentence)):
            if recognized_sentence[i] != correct_sentence[i]:
                recognized_sentence[i] = '*' + recognized_sentence[i]
        print('{:5}: {:60}  {}'.format(video_num, ' '.join(recognized_sentence), ' '.join(correct_sentence)))


def print_sequences(sequences, intro=""):
    print("{}[".format(intro))
    for seq in sequences:
        print("  [")
        for frame in seq:
            print("    {}".format(frame))
        print("  ]")
    print("]")


def print_X_and_L(seq_X, seq_L, intro=""):
    print("{}[".format(intro))
    index = 0
    for length in seq_L:
        print("  [")
        for i in range(length):
            print("    {}".format(seq_X[index]))
            index += 1
        print("  ]")
    print("]")


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

def report_recognize_results(probabilities, guesses, word_list):
    print("\n\nPERFORMED RECOGNITION PASS...")
    word_ID   = 0
    num_hits  = 0
    num_miss  = 0
    
    for word in word_list:
        guess = guesses      [word_ID]
        prob  = probabilities[word_ID][guess]
        print("  Guess for", word, "is", guess, "log. prob:", prob)
        
        if guess == word: num_hits += 1
        else:             num_miss += 1
        word_ID += 1
    
    accuracy = (100 * num_hits) / (num_hits + num_miss)
    print("\nACCURACY: {}%".format(accuracy))

def putHTML(color, msg):
    source = """<font color={}>{}</font><br/>""".format(color, msg)
    return HTML(source)

def feedback(passed, failmsg='', passmsg='Correct!'):
    if passed:
        return putHTML('green', passmsg)
    else:
        return putHTML('red', failmsg)

def getKey(item):
    return item[1]




def combine_sequences(split_index_list, sequences):
    '''
    concatenate sequences referenced in an index list and returns tuple of the new X,lengths

    useful when recombining sequences split using KFold for hmmlearn

    :param split_index_list: a list of indices as created by KFold splitting
    :param sequences: list of feature sequences
    :return: tuple of list, list in format of X,lengths use in hmmlearn
    '''
    sequences_fold = [sequences[idx] for idx in split_index_list]
    X = [item for sublist in sequences_fold for item in sublist]
    lengths = [len(sublist) for sublist in sequences_fold]
    return X, lengths

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



def test_features_tryit(asl):
    print('asl.df sample')
    display(asl.df.head())
    sample = asl.df.ix[98, 1][GROUND_FEATURES].tolist()
    correct = [9, 113, -12, 119]
    failmsg = 'The values returned were not correct.  Expected: {} Found: {}'.format(correct, sample)
    return feedback(sample == correct, failmsg)

def test_std_tryit(df_std):
    print('df_std')
    display(df_std)
    sample = df_std.ix['man-1'][RAW_FEATURES]
    correct = [15.154425, 36.328485, 18.901917, 54.902340]
    failmsg = 'The raw man-1 values returned were not correct.\nExpected: {} for {}'.format(correct, RAW_FEATURES)
    return feedback(np.allclose(sample, correct, .001), failmsg)
