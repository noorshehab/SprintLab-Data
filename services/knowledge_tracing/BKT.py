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
    P(Lj) = P(Lj−1) + P(T) (1 − P(Lj−1)) .
    
    P(Lj |Oj) =1 −((1 − P(T)) [1 − P(Lj−1|Oj−1)] P(G))/(P(G) + (1 − P(S) − P(G)) P(Lj−1|Oj−1))
    when oj = correct
    P(Lj |Oj) = (1 −(1 − P(T)) [1 − P(Lj−1|Oj−1)] (1 − P(G)))/(1 − P(G) − (1 − P(S) − P(G)) P(Lj−1|Oj−1))
    when oj = incorrect.
    note: P(Lj−1|Oj−1) is just the prior here it was also calculated this same way
    
    constraint: P(Lj |Oj),Oj=correct>P(Lj |Oj),Oj=incorrect

    """
    if response:
        p_L=1-((1-improvement)*prior*guess)/(guess+(1-slip-guess)*prior)
    else:
        p_L=1-((1-improvement)*prior*(1-guess))/(1-guess-(1-slip-guess)*prior)
    return p_L

