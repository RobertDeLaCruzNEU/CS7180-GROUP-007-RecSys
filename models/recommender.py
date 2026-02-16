from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np


def ndcg(y: np.ndarray, y_pred: np.ndarray, k: int = 5):
    def dcg(relevances):
        return np.sum([rel / np.log2(i + 2) for i, rel in enumerate(relevances)])
    
    scores = []
    for actual, pred in zip(y, y_pred):
        top_k = pred[:k]
        actual_set = set(actual) if isinstance(actual, (list, np.ndarray)) else {actual}
        relevance_vector = [1 if p in actual_set else 0 for p in top_k]
        actual_dcg = dcg(relevance_vector)
        
        ideal_relevance = [1] * min(len(actual_set), k)
        ideal_dcg = dcg(ideal_relevance)
        
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