def sample_negatives(pos_texts, corpus, num_negatives=1):
    negatives = []
    corpus_set = set(corpus)
    for pos in pos_texts:
        candidates = list(corpus_set - {pos})
        neg_samples = random.sample(candidates, num_negatives)
        negatives.extend(neg_samples)
    return negatives

