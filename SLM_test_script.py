

import random
import math
from my_recognizer import BasicSLM, update_probabilities, report_recognizer_results
import json


def show_guesses(index, all_words, all_probs, all_SLM_probs):
    
    def guess_sort(key):
        return 0 - probs[key]
    
    word        = all_words[index]
    probs       = all_probs[index]
    all_guesses = list(probs.keys())
    all_guesses = sorted(all_guesses, key = guess_sort)
    
    print("\nRecognizer failed for:", word)
    print("Sample probabilities were:")
    
    num_shown = 0
    for guess in all_guesses:
        if num_shown >= 5 and guess != word and guess != all_guesses[-1]: continue
        SLM_prob = all_SLM_probs[index][guess]
        
        print("  {:<16.16} : {:<8.8}  SLM: {:<8.8}".format(guess, str(probs[guess]), str(SLM_prob)))
        num_shown += 1
    
    print("\n")


def normalise_probs(probs):
    max_prob = float("-inf")
    min_prob = float("inf")
    
    for key in probs.keys():
        max_prob = max(max_prob, probs[key])
        min_prob = min(min_prob, probs[key])
    
    prob_range = max_prob - min_prob
    if prob_range == 0: prob_range = 1
    
    for key in probs.keys():
        probs[key] = (probs[key] - min_prob) * 1000. / prob_range




test_SLM = BasicSLM("SLM_data/corpus_sentences.txt", verbose = True)


for gram in range(1, test_SLM.max_grams + 1):
    num_samples = 10
    for i in range(num_samples):
        sample = []
        while len(sample) < gram:
            sample.append(random.choice(test_SLM.all_words))
        likelihood = test_SLM.get_conditional_likelihood(sample)
        print("\nChance of", sample[-1], "after", sample[0:-1], "is", likelihood)

print("\nSampling from fixed word sequence:")

sample_words = ['FISH', 'BOOK', 'VEGETABLE', 'JOHN', 'MARY', 'FUTURE', 'CHINA']

for word in sample_words:
    sample = test_SLM.get_sample(sample_words, sample_words.index(word), word)
    print("  Chance for", sample, "is", test_SLM.get_conditional_likelihood(sample))




all_probs, all_guesses, all_words, all_SLM_probs = [], [], [], []

with open("recognizer_results/raw_results.txt", 'r') as file:
    all_probs, all_guesses, all_words = json.load(file)


print("\nNormalising probabilities...")
for index in range(len(all_words)):
    word  = all_words  [index]
    best  = all_guesses[index]
    probs = all_probs  [index]
    
    if best != word:
        print("  {} mislabelled as {} (index {})".format(word, best, index))
    
    SLM_probs = {}
    for guess in probs.keys():
        sample   = test_SLM.get_sample(all_words, index, guess)
        SLM_prob = test_SLM.get_conditional_likelihood(sample)
        SLM_probs[guess] = SLM_prob
    
    normalise_probs(probs)
    normalise_probs(SLM_probs)
    all_SLM_probs.append(SLM_probs)


show_guesses(2, all_words, all_probs, all_SLM_probs)


"""
acc_before = report_recognizer_results(test_words, test_probs, test_guesses, None, None, None)
test_probs, test_guesses = update_probabilities(test_words, test_probs, test_guesses, test_SLM)
acc_after = report_recognizer_results(test_words, test_probs, test_guesses, None, None, None)

print("\nAccuracy difference: {}%".format(acc_after - acc_before))

with open("recognizer_results/SLM_results.txt", 'w') as file:
    json.dump((test_probs, test_guesses, test_words), file)
"""



