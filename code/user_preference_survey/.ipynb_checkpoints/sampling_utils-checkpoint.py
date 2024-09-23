import itertools
import numpy as np
import random
from collections import Counter

def sample_k_pairs(k: int, mode: str):
    """
    Sample k pairs from the list options according to a probability distribution. Sorts them so that screen minimally changes.
    """

    options = ['html', 'gpt4', 'marker', 'nougat', 'grobid', 'pymupdf', 'pypdf']
    
    # Define the probability distributions based on the mode
    if mode in {'train', 'val'}:
        probs = [0.0, 0.0, 0.25, 0.25, 0.125, 0.25, 0.125]
    else:  # For 'test' or 'val' modes
        probs = [1./12, 1./12, 1./6, 1./6, 1./6, 1./6, 1./6]
    
    # Generate all 2-tuples
    two_tuples = list(itertools.combinations(options, 2))
    two_tuples = [tuple(sorted(t)) for t in two_tuples]
    
    # Create a combined probability distribution for the tuples
    tuple_probs = []
    for t in two_tuples:
        # The probability of a tuple is the product of the individual probabilities
        tuple_prob = probs[options.index(t[0])] * probs[options.index(t[1])]
        tuple_probs.append(tuple_prob)
    
    # Normalize the probabilities (in case they do not sum to 1)
    total_prob = sum(tuple_probs)
    tuple_probs = [p / total_prob for p in tuple_probs]
    
    # Sample k tuples from the two_tuples according to the computed probabilities
    sampled_tuples = random.choices(two_tuples, weights=tuple_probs, k=k)
    
    # Sort the tuples alphabetically
    sampled_tuples.sort()
    
    # Order the tuples to minimize screen changes
    ordered_tuples = order_tuples(sampled_tuples)
    
    return ordered_tuples

def order_tuples(tuples_list):
    """
    Order tuples in a way that makes the binary comparisons as similar as possible
    (e.g. same text on the same side and within consecutive comparison) to reduce exhaustion
    of users
    """
    # Step 1: Flatten the list and count the frequency of each item
    flat_list = [item for t in tuples_list for item in t]
    frequency = Counter(flat_list)
    
    # Step 2: Identify the most frequent item
    most_frequent = frequency.most_common(1)[0][0]
    
    # Step 3: Separate tuples that contain the most frequent item and those that don't
    with_frequent = []
    without_frequent = []
    
    for t in tuples_list:
        if most_frequent in t:
            # Ensure the most frequent item is always in the first position
            if t[1] == most_frequent:
                with_frequent.append((t[1], t[0]))
            else:
                with_frequent.append(t)
        else:
            without_frequent.append(t)
    
    # Step 4: Sort the tuples containing the most frequent item for consecutive appearance
    with_frequent.sort(key=lambda x: x[1])
    
    # Step 5: Combine the ordered tuples
    ordered_list = with_frequent + without_frequent
    
    return ordered_list