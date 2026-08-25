#calculate the semantic similarity of each kc to all the other kcs above 0.7 label them as clustering when setting learning prior these guys get similar error
#create a score based on percentile using the textual features this score biases P(s)
#for questions with under 200 attempts improvement rate is about 1.7 for questions with more than 200 attempts use the actual improvement rate to bias P(r) per item
#p(g) is inverse to p(s) btw 
#reasonable values
#create the dfs of all these values for the test script to use them

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import json
from scipy import stats
load_dotenv()

"""
responses=pd.read_csv(os.getenv('RESPONSES_PROCESSED_PATH'))
learning_deltas=pd.read_csv(os.getenv('chronological_delta_path'))
kc_metadata=pd.read_csv(os.getenv('KC_METADATA_PATH'))"""


def label_clusters(similarity_threshold=0.757):
    """This function takes the kcs and assigns them to clusters based on high semantic similarity"""
    #load the embeddings
    emb_path=os.getenv('KC_EMB_PATH')
    with open(emb_path, 'r') as f:
        kc_embeddings = json.load(f)# question embeddings format like {"kc_id":{embedding dimensions}}
    kc_metadata=pd.read_csv(os.getenv('KC_METADATA_PATH'))

    kcs=list(kc_metadata[kc_metadata['attempted']>=1]['kc_id'])#list of all kc_ids
    embeddings = np.array([kc_embeddings[str(kc)] for kc in kcs])

    #calculate cosine similarity between every kc and all the other kcs in the dataset
    def highest_similarity(kc, similarity_threshold, max_n=12):
        emb = embeddings[int(kc)].reshape(1, -1)
        sim = cosine_similarity(emb, embeddings)[0]

        candidates = [
            (idx, score)
            for idx, score in enumerate(sim)
            if idx != int(kc)
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        selected = []
        sum_sim = 0.0
        for idx, score in candidates:
            new_count = len(selected) + 1
            new_avg = (sum_sim + score) / new_count
            if new_avg < similarity_threshold or new_count > max_n:
                break
            selected.append((idx, score))
            sum_sim += score

        most_similar = [kcs[idx] for idx, _ in selected]
        if len(most_similar) !=0:
            return len(most_similar),sum_sim/len(most_similar), most_similar
        else:
            return len(most_similar),0, most_similar

    df= pd.DataFrame({
        'kc':kcs
    })# i need a df where every row is a kc pair with a high similarity

    df[['n_similar','avg_sim', 'similar_kcs']] = df['kc'].apply(lambda x: pd.Series(highest_similarity(x,similarity_threshold)))

    return df

def set_probs():
    """
    Parameter Constraints:
    1-p(s)>=p(g) for an item
    p(s)+p(g)<1 alternatively
    p(g)<0.3
    p(s)<0.1
    0<p(t)<1
    """
    q_metadata = pd.read_csv(os.getenv('QUESTION_METADATA_PATH'))
    #calculate z scores for important features
    features = ['question_length','num_variables','vocabulary_richness','solution_complexity_x'
                ,'solution_complexity_y','solution_length','num_equations','num_steps']
    
    z_scores = np.array([stats.zscore(q_metadata[feat]) for feat in features])
    #collect the z scores into a big collective score
    collective_score = np.mean(z_scores, axis=0)
    #scale the score so score-min/max-min
    scaled_score = (collective_score - collective_score.min()) / (collective_score.max() - collective_score.min())
    #score * initial p(s) which is 0.1
    p_s = scaled_score * 0.1
    #invert the scaling so abs(score-max/max-min) to get how far the score is from the maximum
    inverted_score = np.abs(collective_score - collective_score.max()) / (collective_score.max() - collective_score.min())
    #score* initial p(g) which is 0.3
    p_g = inverted_score * 0.3
    #return each question id with p(s) and p(g)
    probs=pd.DataFrame(
        q_metadata[['question_id']].copy(),
        columns=['question_id']
    
    )
    probs['p_s'] = p_s
    probs['p_g'] = p_g
    #replace any p(s) or p(g) that is 0 with the threshold value
    probs['p_s'] = probs['p_s'].replace(0, 0.1)
    probs['p_g'] = probs['p_g'].replace(0, 0.3)

    learning_deltas=pd.read_csv(os.getenv('chronological_delta_path'))
    highly_attempted=q_metadata[q_metadata['attempted']>200]['question_id']

    #for the highly attempted questions p_t is abs(learning_delta) from the learning delta df only if learning_delta is -ve
    #if the learning_delta is +ve or the question is not in the well attempted set p(t) to a default 0.017
    probs['p_t'] = learning_deltas.set_index('questions')['learning_delta'].abs().where(
        learning_deltas['questions'].isin(highly_attempted) & (learning_deltas['learning_delta'] < 0), 0.017
    ).reindex(probs['question_id']).fillna(0.017).values

    probs['p_t']=probs['p_t'].replace(0,0.017)
    probs['p_t']=probs['p_t'].replace(1,0.017)

    probs['p_t_constraint']=probs['p_t']<1-probs['p_s']/(1-probs['p_g'])
    
    return probs













