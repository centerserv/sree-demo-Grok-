def update_trust(prior_trust, accuracy):
    """Update trust score using Bayesian-inspired method."""
    likelihood = accuracy
    trust = (prior_trust * likelihood) / (prior_trust * likelihood + (1 - prior_trust) * (1 - likelihood))
    return trust