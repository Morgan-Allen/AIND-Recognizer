

import random
import math
from my_recognizer import BasicSLM, update_probabilities, report_recognizer_results
import json



test_SLM = BasicSLM("SLM_data/corpus_sentences.txt", verbose = False)

        
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
recent_words = []

for word in sample_words:
    sample = test_SLM.get_sample(recent_words, word)
    print("  Chance for", sample, "is", test_SLM.get_conditional_likelihood(sample))
    recent_words.append(word)


test_probs, test_guesses, test_words, recent_words = [], [], [], []

with open("recognizer_results/raw_results.txt", 'r') as file:
    test_probs, test_guesses, test_words = json.load(file)


for index in range(len(test_words)):
    word  = test_words  [index]
    best  = test_guesses[index]
    probs = test_probs  [index]
    
    if best == word:
        recent_words.append(best)
        continue
    
    def guess_sort(key):
        return 0 - probs[key]
    
    all_guesses = list(probs.keys())
    all_guesses = sorted(all_guesses, key = guess_sort)
    
    print("\nRecognizer failed for:", word, "guessed:", best)
    print("Recent words would be:", test_SLM.get_sample(recent_words, word))
    print("Example probabilities were:")
    
    num_shown = 0
    for guess in all_guesses:
        if num_shown >= 5 and guess != word and guess != all_guesses[-1]: continue
        
        sample   = test_SLM.get_sample(recent_words, guess)
        SLM_prob = math.log(test_SLM.get_conditional_likelihood(sample))
        
        print("  {:<12} : {:<8.8}  SLM: {:<8.8}".format(guess, str(probs[guess]), str(SLM_prob)))
        num_shown += 1
    
    print("\n")
    recent_words.append(best)
    break


"""
acc_before = report_recognizer_results(test_words, test_probs, test_guesses, None, None, None)
test_probs, test_guesses = update_probabilities(test_words, test_probs, test_guesses, test_SLM)
acc_after = report_recognizer_results(test_words, test_probs, test_guesses, None, None, None)

print("\nAccuracy difference: {}%".format(acc_after - acc_before))

with open("recognizer_results/SLM_results.txt", 'w') as file:
    json.dump((test_probs, test_guesses, test_words), file)
"""



