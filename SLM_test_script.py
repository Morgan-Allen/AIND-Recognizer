

import random
from my_recognizer import BasicSLM



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