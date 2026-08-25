import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from services.Interfaces import Component
from services.log_setup import get_logger
from services.question_processing.embedding_service import embedding_service
from services.question_processing import feature_derivation
from services.question_processing import prob_calcs

log=get_logger('question_processing_service')

#default clustering threshold / neighbor cap, matching data_preprocess.label_clusters
SIMILARITY_THRESHOLD = 0.757
MAX_NEIGHBORS = 12


class question_processing_service(Component):
    """Embeds skills (from their text) to build the similar-skills map, and
    derives features + BKT probabilities for questions.
    """

    def __init__(self, embedder=None, similarity_threshold=SIMILARITY_THRESHOLD,
                 max_neighbors=MAX_NEIGHBORS, text_col='Question_Text', solutions_col='Correct_Answer_Content'):
        self.mediator = None
        self.embedder = embedder or embedding_service()
        self.similarity_threshold = similarity_threshold
        self.max_neighbors = max_neighbors
        self.text_col = text_col
        self.solutions_col = solutions_col
        self.skill_embeddings = {}   # {skill_id: ndarray}
        self.similar_skills_map = {}  # {skill_id: [similar_skill_id, ...]}

    #--------------------------------------------------------------------------
    # embedding
    #--------------------------------------------------------------------------
    def embed_skills(self, skill_texts=None, skill_embeddings=None):
        """Build skill embeddings.

        skill_texts: {skill_id: text} or list of (skill_id, text) — embedded
        via the embedding service to produce {skill_id: ndarray}.
        skill_embeddings: optional precomputed {skill_id: ndarray} loaded from
        a file (similar to KC_EMB_PATH); if given it skips the embedder.
        """
        if skill_embeddings:
            self.skill_embeddings = {
                k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in skill_embeddings.items()
            }
            return self.skill_embeddings

        items = skill_texts if skill_texts is not None else []
        if isinstance(items, dict):
            items = list(items.items())
        if not items:
            log.warning('embed_skills: no skill text provided; '
                  'supply skill_texts or precomputed skill_embeddings')
            return self.skill_embeddings

        #preserve id type: skill ids may be ints or strings like 'KC-BIO-01'
        ids = [int(i) if isinstance(i, float) and float(i).is_integer() else i
               for i, _ in items]
        texts = [str(t) for _, t in items]

        cached, to_embed, to_embed_ids = {}, [], []
        for sid, text in zip(ids, texts):
            vec = self.embedder.get_cached(sid)
            if vec is not None:
                cached[sid] = vec
            else:
                to_embed.append(text)
                to_embed_ids.append(sid)

        new_vectors = self.embedder.embed(to_embed, ids=to_embed_ids) if to_embed else {}
        all_vecs = {**cached, **new_vectors}

        for sid in ids:
            vec = all_vecs.get(sid)
            if vec is not None:
                self.skill_embeddings[sid] = np.asarray(vec, dtype=np.float64)

        if new_vectors:
            self.embedder.save_cache(new_vectors)

        log.info('embedded %d skills (cached=%d)',len(all_vecs),len(cached))
        return self.skill_embeddings

    #--------------------------------------------------------------------------
    # similar-skills map (same clustering as data_preprocess.label_clusters)
    #--------------------------------------------------------------------------
    def build_similar_skills_map(self, skill_embeddings=None):
        """Cluster skill embeddings by cosine similarity (greedy, average above threshold)."""
        if skill_embeddings is not None:
            self.skill_embeddings = {
                k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in skill_embeddings.items()
            }
        skills = sorted(self.skill_embeddings.keys())
        if not skills:
            return {}
        emb_matrix = np.vstack([self.skill_embeddings[s] for s in skills])
        sims = cosine_similarity(emb_matrix)

        def highest_similarity(idx):
            candidates = [(j, sims[idx, j]) for j in range(len(skills)) if j != idx]
            candidates.sort(key=lambda x: x[1], reverse=True)
            selected, sum_sim = [], 0.0
            for j, score in candidates:
                new_count = len(selected) + 1
                new_avg = (sum_sim + score) / new_count
                if new_avg < self.similarity_threshold or new_count > self.max_neighbors:
                    break
                selected.append(j)
                sum_sim += score
            return [skills[j] for j in selected]

        self.similar_skills_map = {
            sid: highest_similarity(i) for i, sid in enumerate(skills)
        }
        log.info('similar-skills map: %d skills '
              '(threshold=%s, max_n=%d)',len(self.similar_skills_map),
              self.similarity_threshold,self.max_neighbors)
        return self.similar_skills_map

    #--------------------------------------------------------------------------
    # feature + probability derivation
    #--------------------------------------------------------------------------
    def derive_question_attributes(self, metadata_df, question_skills=None):
        """Run all attribute + probability derivations on question metadata."""
        self._question_skills = question_skills or {}
        attrs = feature_derivation.derive_features(
            metadata_df,
            text_col=self.text_col,
            solutions_col=self.solutions_col,
        )
        probs = prob_calcs.compute_probs(attrs.copy())
        enriched = pd.merge(attrs, probs[['question_id', 'p_s', 'p_g', 'p_t']], on='question_id', how='left')
        enriched['p_t'] = enriched['p_t'].fillna(prob_calcs.DEFAULT_P_T)
        return enriched

    def register_skills(self, skills_map=None):
        """Push the similar-skills map into the data service."""
        smap = skills_map or self.similar_skills_map
        for sid, similar in smap.items():
            self.mediator.request( {
                'type': 'add_skill', 'skill_id': sid, 'similar_skills': similar,
            })
        return list(smap.keys())

    #--------------------------------------------------------------------------
    # end-to-end convenience
    #--------------------------------------------------------------------------
    def process(self, metadata_df, questions_json=None, skill_texts=None,
                skill_embeddings=None, question_skills=None):
        """High-level entry: skill embeddings + skills map + feature/prob frame.

        Pass skill_texts ({skill_id: text} or [(skill_id, text), ...]) to embed
        skills via the service, or skill_embeddings ({skill_id: ndarray}) to use
        precomputed vectors.
        """
        if question_skills is None:
            question_skills = self._read_question_skills(metadata_df)
        self._question_skills = question_skills

        self.embed_skills(skill_texts=skill_texts, skill_embeddings=skill_embeddings)
        self.build_similar_skills_map()

        return self.derive_question_attributes(metadata_df)

    def _read_question_skills(self, df):
        """Extract {question_id: [skill_ids]} from a skill/cluster column."""
        id_col = next((c for c in ['question_id', 'Question_ID'] if c in df.columns), None)
        if id_col is None:
            return {}
        col = next((c for c in ['skill_ids', 'Skill_Cluster_ID', 'kc_ids'] if c in df.columns), None)
        if col is None:
            return {}
        out = {}
        for qid, raw in zip(df[id_col], df[col]):
            skills = self._parse_skill_list(raw)
            out[str(qid)] = skills
        return out

    @staticmethod
    def _parse_skill_list(raw):
        import ast
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return []
        if isinstance(raw, str):
            try:
                parsed = ast.literal_eval(raw.strip())
            except (ValueError, SyntaxError):
                return [raw]
            return parsed if isinstance(parsed, list) else [parsed]
        if isinstance(raw, (list, tuple)):
            return list(raw)
        return [raw]