def update_trust(prior_trust, accuracy, decay=0.8):
    """Update trust with decay for declining accuracy."""
    likelihood = accuracy
    if accuracy < prior_trust:
        likelihood *= decay
    trust = (prior_trust * likelihood) / (prior_trust * likelihood + (1 - prior_trust) * (1 - likelihood))
    return max(0.5, min(trust, 0.97))
