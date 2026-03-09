from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np

import numpy as np

def ndcg(y: np.ndarray, y_pred: np.ndarray, k: int = 5):
    def dcg(relevances):
        relevances = np.asarray(relevances, dtype=float)
        if relevances.size:
            return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
        return 0.0

    scores = []
    for actual, pred in zip(y, y_pred):
        relevance_map = {item_id: len(actual) - i for i, item_id in enumerate(actual)}
        
        top_k = pred[:k]
        
        relevance_vector = [relevance_map.get(item_id, 0) for item_id in top_k]
        actual_dcg = dcg(relevance_vector)
        
        ideal_relevances = sorted(relevance_map.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal_relevances)
        
        if ideal_dcg > 0:
            scores.append(actual_dcg / ideal_dcg)
        else:
            scores.append(0.0)
            
    return np.mean(scores)


class Recommender(NearestNeighbors):
    def __init__(
            self,
            n_neighbors:int=5,
            radius:float=1.0,
            algorithm:str='auto',
            leaf_size:int=30,
            metric:str='minkowski',
            p:int=2,
            metric_params=None,
            n_jobs=None
        ):
        super().__init__(n_neighbors=n_neighbors, 
                         radius=radius, 
                         algorithm=algorithm, 
                         leaf_size=leaf_size, 
                         metric=metric, 
                         p=p, metric_params=metric_params, 
                         n_jobs=n_jobs)
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        super().fit(X)
        self.y = y
        return self

    def predict(self, X: np.ndarray):
        _, indices = self.kneighbors(X)
        if self.y is None:
            return indices
            
        results = []
        for row in indices:
            results.append([self.y[i] for i in row])
        return np.array(results)
    
    def score(self, X: np.ndarray, y: np.ndarray, k: int = 5):
        y_pred = self.predict(X)
        return ndcg(y, y_pred, k=k)