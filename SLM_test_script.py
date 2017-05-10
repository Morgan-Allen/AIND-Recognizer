

import random
from my_recognizer import BasicSLM, update_probabilities, report_recognizer_results
import json



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
recent_words = []

for word in sample_words:
    sample = test_SLM.get_sample(recent_words, word)
    print("  Chance for", sample, "is", test_SLM.get_conditional_likelihood(sample))
    recent_words.append(word)


with open("recognizer_results/raw_results.txt", 'r') as file:
    
    test_probs, test_guesses, test_words = json.load(file)
    acc_before = report_recognizer_results(test_words, test_probs, test_guesses, None, None, None)
    
    test_probs, test_guesses = update_probabilities(test_words, test_probs, test_guesses, test_SLM)
    acc_after = report_recognizer_results(test_words, test_probs, test_guesses, None, None, None)
    
    print("\nAccuracy difference: {}%".format(acc_after - acc_before))

with open("recognizer_results/SLM_results.txt", 'w') as file:
    json.dump((test_probs, test_guesses, test_words), file)


