#the functions that take the prior probabilities and calculate P(L) and P(C)
#their input is the set of student and item parameters and 1 or 0 for correct and incorrect responses
#output is the adjusted per student prior and the prediction for the next item for the skill

def next_response(prior,guess,slip):
    """
    P(Cj) = P(G) (1 − P(Lj)) + (1 − P(S)) P(Lj) .
    constraint: 0<p(Cj)<1
    """
    p_C= guess*(1-prior)+(1-slip)*prior
    next_response=0
    if p_C>0.5:
        next_response=1
    return p_C,next_response

def update_prior(prior,guess,slip,improvement,response):
    """
    Standard Bayesian BKT update.

    1. learning transition: P*(L) = P(L) + P(T) (1 - P(L))
    2. evidence update via Bayes' rule on the response:

       correct:   P(L|C) = P*(L)(1-P(S)) / (P*(L)(1-P(S)) + (1-P*(L))P(G))
       incorrect: P(L|W) = P*(L)P(S)     / (P*(L)P(S)     + (1-P*(L))(1-P(G)))

    Guarantees (for 0<P(S),P(G)<1):
      - a correct answer raises the prior, a wrong answer lowers it
      - P(L|correct) >= P(L|wrong) from any identical starting state
      - the result stays strictly inside (0, 1); the engine additionally
        clamps to [0.01, 0.99] so stored priors never reach the bounds
    """
    learned = prior + improvement * (1 - prior)

    if response:
        numerator = learned * (1 - slip)
        denominator = numerator + (1 - learned) * guess
    else:
        numerator = learned * slip
        denominator = numerator + (1 - learned) * (1 - guess)

    if denominator <= 0:
        return prior
    return numerator / denominator
