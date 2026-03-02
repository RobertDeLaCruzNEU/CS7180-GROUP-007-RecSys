import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class UserFeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Create creator levels
        X['new_creator_level'] = 'Lurker'
        X.loc[X['is_video_author'] == 1, 'new_creator_level'] = 'Creator'
        X.loc[X['is_live_streamer'] == 1, 'new_creator_level'] = 'Streamer'

        # Binning register days
        X['register_days_range'] = pd.cut(
            X['register_days'], 
            bins=[0, 180, 365, 365*2, 365*3, 365*4, 365*5, 365*6, 365*7, np.inf], 
            labels=['0-180 days', '181-365 days', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-6 years', '6-7 years', '7+ years']
        )

        # Binning follow counts
        X['follow_user_num_range'] = pd.cut(
            X['follow_user_num'], 
            bins=[0, 10, 50, 100, 250, 500, np.inf], 
            labels=['(0,10]', '(10,50]', '(50,100]', '(100,250]', '(250, 500]', '500+']
        )

        # Binning fan counts
        X['fans_user_num_range'] = pd.cut(
            X['fans_user_num'], 
            bins=[0, 10, 100, 1000, np.inf], 
            labels=['[0-10)', '[11-100)', '[101-1k)', '1k+']
        )

        # Clean engagement levels
        X['user_active_degree'] = X['user_active_degree'].apply(lambda x: str(x).replace("_", " ").capitalize())
        X.loc[X['user_active_degree'].isin(['High active', 'Full active']), 'user_active_degree'] = 'High/Full active'

        return X
    
CATEGORICAL_COLS = [
    'user_active_degree', 
    'follow_user_num_range', 
    'fans_user_num_range', 
    'register_days_range', 
    'new_creator_level'
]

NUMERICAL_COLS = [
    'follow_user_num', 
    'fans_user_num', 
    'friend_user_num', 
    'register_days'
]

UserFeaturePreprocessor = Pipeline(steps=[
    ('engineering', UserFeatureEngineering()),
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), CATEGORICAL_COLS),
            ('num', 'passthrough', NUMERICAL_COLS)
        ]
    ))
])