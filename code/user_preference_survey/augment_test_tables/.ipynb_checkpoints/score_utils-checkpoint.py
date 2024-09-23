import pandas as pd
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from rapidfuzz import fuzz

def calculate_bleu(reference: str, hypothesis: str) -> float:
    """
    Compute BLEU score
    """
    reference_tokens = nltk.word_tokenize(reference)
    hypothesis_tokens = nltk.word_tokenize(hypothesis)
    
    # Use SmoothingFunction to handle cases with zero counts in higher-order n-grams
    smoothing_function = SmoothingFunction().method1
    
    return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing_function)

def calculate_meteor(reference: str, hypothesis: str) -> float:
    """
    Compute METEOR score
    """
    reference_tokens = nltk.word_tokenize(reference)
    hypothesis_tokens = nltk.word_tokenize(hypothesis)
    
    # METEOR expects a list of reference sentences
    reference_tokens_list = [reference_tokens]
    
    return meteor_score(reference_tokens_list, hypothesis_tokens)

def calculate_rouge(reference: str, hypothesis: str) -> float:
    """
    Compute ROUGE score
    """
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    
    return scores['rouge2'].fmeasure

def calculate_car(reference: str, hypothesis: str) -> float:
    """
    Compute character accuracy rate (CAR)
    """
    return fuzz.ratio(reference, hypothesis) / 100.0