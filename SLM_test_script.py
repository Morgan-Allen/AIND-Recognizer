

import random
from my_recognizer import (
    BasicSLM, get_SLM_probs, normalise_and_combine, report_recognizer_results
)
import json


def show_guesses(index, all_words, all_probs, all_SLM_probs, test_SLM):
    
    def guess_sort(key):
        return 0 - probs[key]
    
    def SLM_sort(key):
        return 0 - SLM_probs[key]
    
    word        = all_words    [index]
    probs       = all_probs    [index]
    SLM_probs   = all_SLM_probs[index]
    all_guesses = list(probs.keys())
    all_guesses = sorted(all_guesses, key = guess_sort)
    SLM_guesses = list(probs.keys())
    SLM_guesses = sorted(SLM_guesses, key = SLM_sort)
    SLM_best    = SLM_guesses[0]
    SLM_worst   = SLM_guesses[-1]
    sample      = test_SLM.get_sample(all_words, index, word)
    
    print("\nRecognizer failed for:", word)
    print("Sample sequence would be:", sample)
    print("Sample probabilities were:")
    
    num_shown = 0
    for guess in all_guesses:
        if num_shown >= 5 and guess != word and guess != all_guesses[-1] and guess != SLM_best and guess != SLM_worst: continue
        
        print("  {:<16.16} : {:<8.8}  SLM: {:<8.8}".format(guess, str(probs[guess]), str(SLM_probs[guess])))
        num_shown += 1
    
    print("\n")




test_SLM = BasicSLM("SLM_data/corpus_sentences.txt", max_grams = 3, verbose = True)


for gram in range(1, test_SLM.max_grams + 1):
    num_samples = 10
    for i in range(num_samples):
        sample = []
        while len(sample) < gram:
            sample.append(random.choice(test_SLM.all_words))
        likelihood = test_SLM.get_conditional_prob(sample)
        print("\nChance of", sample[-1], "after", sample[0:-1], "is", likelihood)

print("\nSampling from fixed word sequence:")

sample_words = ['FISH', 'BOOK', 'VEGETABLE', 'JOHN', 'MARY', 'FUTURE', 'CHINA']

for word in sample_words:
    sample = test_SLM.get_sample(sample_words, sample_words.index(word), word)
    print("  Chance for", sample, "is", test_SLM.get_max_prob(sample))




all_probs, all_guesses, all_words = [], [], []

with open("recognizer_results/raw_results.txt", 'r') as file:
    all_probs, all_guesses, all_words = json.load(file)

acc_before = report_recognizer_results(all_words, all_probs, all_guesses, None, None, None)

all_SLM_probs = get_SLM_probs(all_words, all_probs, test_SLM)

print("\n\nChecking for all specific mismatches...")
for index in range(len(all_words)):
    word  = all_words  [index]
    best  = all_guesses[index]
    if best != word:
        print("  {} mislabelled as {} (index {})".format(word, best, index))

show_guesses(2 , all_words, all_probs, all_SLM_probs, test_SLM)
show_guesses(8 , all_words, all_probs, all_SLM_probs, test_SLM)
show_guesses(12, all_words, all_probs, all_SLM_probs, test_SLM)
show_guesses(18, all_words, all_probs, all_SLM_probs, test_SLM)


new_probs, new_guesses = normalise_and_combine(all_words, all_probs, all_SLM_probs, all_guesses, 1)
acc_after = report_recognizer_results(all_words, new_probs, new_guesses, None, None, None)


print("\nAccuracy difference: {}%".format(acc_after - acc_before))

with open("recognizer_results/SLM_results.txt", 'w') as file:
    json.dump((new_probs, new_guesses, all_words), file)






