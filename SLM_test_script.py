

import random
from my_recognizer import (
    BasicSLM, get_SLM_probs, scale_and_combine, report_recognizer_results
)
import json


def show_guesses(index, all_guesses, all_probs, all_SLM_probs, real_words):
    
    def guess_sort(key):
        return 0 - probs[key]
    
    def SLM_sort(key):
        return 0 - SLM_probs[key]
    
    recent_words = None
    if index <= 3: recent_words = real_words[0:index]
    else: recent_words = real_words[index - 3:index]
    
    recent_guesses = None
    if index <= 3: recent_guesses = all_guesses[0:index]
    else: recent_guesses = all_guesses[index - 3:index]
    
    word         = all_words    [index]
    probs        = all_probs    [index]
    SLM_probs    = all_SLM_probs[index]
    sort_guesses = list(probs.keys())
    sort_guesses = sorted(sort_guesses, key = guess_sort)
    SLM_guesses  = list(probs.keys())
    SLM_guesses  = sorted(SLM_guesses, key = SLM_sort)
    SLM_best     = SLM_guesses[0]
    SLM_worst    = SLM_guesses[-1]
    
    print("\nRecognizer failed for: {}, (index {})".format(word, index))
    print("Recent words were:  ", recent_words  )
    print("Recent guesses were:", recent_guesses)
    print("Sample probabilities were:")
    
    num_shown = 0
    for guess in sort_guesses:
        if num_shown >= 5 and guess != word and guess != all_guesses[-1] and guess != SLM_best and guess != SLM_worst: continue
        
        print("  {:<16.16} : {:<8.8}  SLM: {:<8.8}".format(guess, str(probs[guess]), str(SLM_probs[guess])))
        num_shown += 1
    
    print("\n")



test_SLM = BasicSLM("SLM_data/corpus_sentences.txt", max_grams = 3, verbose = True)

all_probs, all_guesses, all_words = [], [], []

with open("recognizer_results/raw_results.txt", 'r') as file:
    all_probs, all_guesses, all_words = json.load(file)

acc_before = report_recognizer_results(all_words, all_probs, all_guesses, None, None, None)

new_probs, new_guesses = scale_and_combine(all_guesses, all_probs, test_SLM, 20)

acc_after = report_recognizer_results(all_words, new_probs, new_guesses, None, None, None)


num_better = 0
num_worse  = 0

for index in range(len(all_words)):
    word      = all_words  [index]
    best_old  = all_guesses[index]
    best_new  = new_guesses[index]
    
    if best_new == word and best_new != best_old: num_better += 1
    if best_old == word and best_new != best_old: num_worse  += 1

print("\nAccuracy difference: {}%".format(acc_after - acc_before))
print("  New successes: {}".format(num_better))
print("  New failures:  {}".format(num_worse ))

with open("recognizer_results/SLM_results.txt", 'w') as file:
    json.dump((new_probs, new_guesses, all_words), file)








