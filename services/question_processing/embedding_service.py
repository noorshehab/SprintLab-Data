import hashlib
import json
import os
import numpy as np

DEFAULT_MODEL = 'Qwen/Qwen3-Embedding-0.6B'


class embedding_service:
    """Embed question text with Qwen3-Embedding-0.6B.

    Uses a lazy import of torch/transformers so the rest of the pipeline works
    even before the model is installed. When torch/transformers are unavailable,
    falls back to a deterministic feature-hash embedding so ingestion/vector-store
    flows can be exercised; set USE_EMBEDDING_FALLBACK=0 to force the real model.
    """

    def __init__(self, model_name=None, cache_path=None, use_fallback=None):
        self.model_name = model_name or os.getenv('EMBEDDING_MODEL', DEFAULT_MODEL)
        self.base_cache_path = cache_path or os.getenv('EMBEDDING_CACHE_PATH')
        if use_fallback is None:
            env = os.getenv('EMBEDDING_FALLBACK', '1')
            use_fallback = env not in ('0', 'false', 'False')
        self.use_fallback = use_fallback
        self._tokenizer = None
        self._model = None
        self._device = None
        self._dim = None
        #mode-specific suffix so real and fallback vectors never mix in one file
        self.cache_path = self._mode_cache_path()
        self._cache = self._load_cache()

    def _mode_cache_path(self):
        if not self.base_cache_path:
            return None
        root, ext = os.path.splitext(self.base_cache_path)
        mode = 'fallback' if self.use_fallback else 'qwen'
        return f'{root}_{mode}{ext}'

    #--------------------------------------------------------------------------
    # model lifecycle
    #--------------------------------------------------------------------------
    def _load_model(self):
        if self._model is not None:
            return True
        if self.use_fallback:
            #EMBEDDING_FALLBACK=1 explicitly forces the fast deterministic
            #embeddings (impractical to run cpu Qwen on the full corpus)
            print('[embedding] EMBEDDING_FALLBACK=1 -> using deterministic fallback embeddings')
            return False
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            print('[embedding] transformers/torch not installed -> using fallback embeddings')
            return False

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self._model.eval()
        if torch.cuda.is_available():
            import torch as _t
            self._device = _t.device('cuda')
            self._model.to(self._device)
        else:
            self._device = self._model.device
        self._dim = self._model.config.hidden_size
        print(f'[embedding] loaded {self.model_name} on {self._device} (dim={self._dim})')
        return True

    def available(self):
        """True if the real (Qwen3) model can produce embeddings right now."""
        try:
            return self._load_model()
        except Exception:
            return False

    #--------------------------------------------------------------------------
    # embedding
    #--------------------------------------------------------------------------
    def embed(self, texts, ids=None):
        """Embed a list of texts. Returns {id or index: ndarray}."""
        texts = list(texts)
        ids = list(ids) if ids is not None else list(range(len(texts)))

        if self._load_model():
            vectors = self._embed_real(texts)
        else:
            vectors = self._embed_fallback(texts)

        out = {}
        for qid, vec in zip(ids, vectors):
            out[qid] = vec
        return out

    def _embed_real(self, texts, batch_size=32):
        import torch
        import torch.nn.functional as F

        embeddings = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            seqs = self._tokenizer(
                chunk, padding=True, truncation=True, max_length=1024, return_tensors='pt'
            )
            with torch.no_grad():
                outputs = self._model(**seqs)
            mask = seqs['attention_mask'].unsqueeze(-1).float()
            hidden = outputs[0]
            mean_pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                mean_pooled = outputs.pooler_output
            vecs = F.normalize(mean_pooled, p=2, dim=1).cpu().numpy()
            embeddings.extend(v.astype(np.float64) for v in vecs)
        return embeddings

    def _embed_fallback(self, texts):
        """Deterministic feature-hash embedding (char n-grams + word tokens)."""
        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text, dim=256):
        vec = np.zeros(dim, dtype=np.float64)
        t = str(text).lower()
        tokens = _ngrams(t, 2) + _ngrams(t, 3) + t.split()
        for tok in tokens:
            h = int(hashlib.blake2b(tok.encode('utf-8'), digest_size=8).hexdigest(), 16)
            vec[h % dim] += 1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    #--------------------------------------------------------------------------
    # caching
    #--------------------------------------------------------------------------
    def _load_cache(self):
        if not self.cache_path or not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def get_cached(self, qid):
        raw = self._cache.get(str(qid))
        return np.asarray(raw, dtype=np.float64) if raw is not None else None

    def save_cache(self, new_entries):
        if not self.cache_path:
            return
        for k, v in new_entries.items():
            self._cache[str(k)] = np.asarray(v).tolist()
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self._cache, f)


def _ngrams(text, n):
    return [text[i:i + n] for i in range(max(0, len(text) - n + 1))]