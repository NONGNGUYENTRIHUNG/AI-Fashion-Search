"""Frozen CLIP retrieval; optional supervised classifier trained on its embeddings."""
from pathlib import Path
import threading
from urllib.parse import quote
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = 'sentence-transformers/clip-ViT-B-32'
class Engine:
    def __init__(self):
        self._model = None
        self._lock = threading.RLock()
        self.paths, self.vectors, self.signature = [], None, None

    def encode(self, inputs):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(MODEL_NAME)
            return np.asarray(self._model.encode(inputs, normalize_embeddings=True, show_progress_bar=False))

    def search(self, text='', image=None, limit=12):
        paths = sorted(p for p in (ROOT/'data/catalog').iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
        if not paths:
            return []
        signature = tuple((p.name,p.stat().st_mtime_ns,p.stat().st_size) for p in paths)
        with self._lock:
            if signature != self.signature:
                vectors=[]
                for p in paths:
                    with Image.open(p) as im:
                        vectors.append(self.encode([im.convert('RGB')])[0])
                self.paths, self.vectors, self.signature = paths,np.stack(vectors),signature
            queries=[]
            if image is not None:
                queries.append(self.encode([image])[0])
            if text.strip():
                queries.append(self.encode([text.strip()])[0])
            query=np.mean(queries,axis=0)
            query=query/max(float(np.linalg.norm(query)),1e-12)
            scores=self.vectors@query
            return [dict(id=self.paths[i].name,title=self.paths[i].stem.replace('_',' ').replace('-',' '),
                         image_url='/catalog/'+quote(self.paths[i].name),similarity=round(float(scores[i]),4))
                    for i in np.argsort(-scores)[:limit]]

    def classify(self,image):
        path=ROOT/'models/classifier.joblib'
        if not path.exists():
            return None
        import joblib
        bundle=joblib.load(path)  # Only load your own locally trained artifact.
        if bundle['encoder'] != MODEL_NAME:
            raise ValueError('Classifier encoder does not match search encoder')
        probs=bundle['model'].predict_proba(self.encode([image]))[0]
        labels=bundle['model'].classes_
        return sorted([{'label':str(c),'probability':round(float(p),4)} for c,p in zip(labels,probs)],key=lambda x:-x['probability'])[:3]
