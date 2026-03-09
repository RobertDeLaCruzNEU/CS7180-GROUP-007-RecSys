import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class InteractionFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.field_map = {
            'user_id': 'user_id', 
            'video_id': 'video_id',
            'c0': 'User component 1', 
            'c1': 'User component 2',
            'v1': 'Video component 1', 
            'v2': 'Video component 2', 
            'v3': 'Video component 3', 
            'cluster_user': 'User segment', 
            'cluster_video': 'Video segment', 
            'date': 'Date', 
            'is_like': 'User liked',
            'long_view': 'Viewed most', 
            'is_profile_enter': "Viewed creator's profile"
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        users, videos, logs = X
        X = users.merge(logs, on='user_id', how='right').merge(videos, on='video_id', how='left', suffixes=('_user','_video'))
        X = X.rename(columns=self.field_map)
        return X
    
    def fit_transform(self, X, y = None, **fit_params):
        return super().fit_transform(X, y, **fit_params)
    


# Configuration for the Pipeline
CATEGORICALS = [
    'User segment', 
    'Video segment' 
]

NUMERICALS = [
    'User component 1',	
    'User component 2',	
    'Video component 1',	
    'Video component 2',	
    'Video component 3'
]

InteractionFeaturePreprocessor = Pipeline(steps=[
    ('engineering', InteractionFeatureEngineering()),
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32), CATEGORICALS),
            ('num', 'passthrough', NUMERICALS)
        ]
    ))
])
