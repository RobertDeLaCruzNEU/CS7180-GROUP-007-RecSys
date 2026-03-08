import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class VideoFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.top_creators = {
            'Top 20': [ 162618, 5921989, 1127691, 5294188, 6239090, 4642129, 5097930, 3967151, 5623243, 7497066, 6599534, 3952568, 5536043, 4927033, 6770796, 7136643, 5117904, 3529930, 7055822, 4827122],
            'Top 20-40': [3376439, 5388905, 7311352, 8428816, 5363056, 5199230, 8383105, 7795287, 3503742, 1500262, 8203851, 4494478, 6710390, 78931, 420876, 5507915, 7325453, 4018420, 8443599, 194116],
            'Top 40-60': [7586832, 4052956, 5115904, 8541212, 1864991, 2048237, 7482152, 7490392,  833612, 7294138, 5838090, 1039764, 7642885, 106137, 7532107,  465827,  909536, 3605982, 7542571, 8678],
            'Top 60-80': [7376629, 1704409, 5132825, 6432207, 4921585, 4869820, 6689331, 5309459, 6000758, 7898710, 7897411, 6853179, 6004355, 7997883, 2456727, 6780903, 7163588, 5236343, 5508966, 4256925],
            'Top 80-100': [  59454, 7400255, 4969630, 7350552, 7348269, 8277168, 7699565, 7402091, 7631453, 7564962, 7990110, 1508379, 5198513, 5204448, 642953, 6234226, 5040662, 6643294, 5969836, 5643385]
        }
        self.top_songs = {
            'None': [0],
            'Top 20 song': [9107419068, 6201083918, 8953211090, 9156967270, 9056382190, 9084472638, 8911046447, 8759931155, 9212341703, 9001113574, 8514002961, 5128196645, 8871829651,      39883, 5128386736, 8928004161, 8873596998, 9090448530, 8312074764],
            'Top 20-40': [9179209248, 4096626583, 8887918349, 9108660117, 8968932964, 9001314800, 9022650002, 9119097210, 6199377620, 8147298784, 9070399849, 6200169289, 9066327392, 8875890895, 3095969283, 5128480679, 9179004717, 7942400530, 8858015130, 3365310207],
            'Top 40-60': [5127670285, 6821917069, 8918965475, 9190020176, 9118882131, 9156568021, 3151246603, 8857332209, 8847892177, 8525244297, 8851473258, 9013366940, 5127344928, 8033050120, 8084903889, 4045710830, 8863754454, 9144593462, 8991533641, 8465852089],
            'Top 60-80': [9051431556, 8802023739, 9139215406, 8147112553, 9150424395, 8672574101, 8929429655, 8003355658, 5128097221, 8867093581, 9162494612, 8977948330, 9053668018, 4751083022, 8146716262, 8862696427,     116759, 8614173079, 2167228536, 9136006423],
            'Top 80-100': [9146223172, 9034717539, 7751660740, 7097343593, 8836684413, 7090006253, 9035243807, 6592739598, 8862696449, 4720441703, 8480212178, 8000470277, 8399959119, 8931826945, 7655044588, 7688522061, 8981513909, 8871639034, 9138314412, 8514898376]
        }
        self.field_map = {
            "video_id": "video_id",
            "video_type": "Type of this video.",
            "upload_type": "The upload type of this video.",
            "visible_status": "The visible state of this video on the APP now.",
            "video_duration": "The time duration of this video (in millisecond).",
            "music_type": "Background music type of this video.",
            "tag": "Video hashtag.",

            "video_display_size": "The display area of this video.",
            "video_duration_range": "The duration range of this video.",
            "video_display_size_range": "The display area range of this video.",
            "video_age": "The age of this video (in days).",
            "video_age_range": "The age range of this video.",
            "author_class": "The class of the author of this video.",
            "song_rank": "The rank of the background music of this video in the music library.",
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Calculate reference date for video age during fit
        self.max_upload_dt_ = pd.to_datetime(X['upload_dt'], unit='s', errors='coerce').max()
        X['upload_dt'] = X.upload_dt.fillna(self.max_upload_dt_.timestamp())
        
        X['server_height'] = X['server_height'].fillna(X.server_height.mean())
        X['server_width']  = X['server_width'].fillna(X.server_width.mean())

        X['visible_status'] = X['visible_status'].fillna(X.visible_status.mode()[0])
        X['music_type'] = X['music_type'].fillna(-1)

        # 1. Video duration range
        X['video_duration'] = X['video_duration'].fillna(X.video_duration.mean())
        X['video_duration_range'] = pd.cut(
            X['video_duration'], 
            bins=[0, 3e4, 6e4, 1.2e5, 1.8e5, 2.4e5, 3e5, 3.6e5, 4.2e5, np.inf], 
            labels=['0-30s', '30-60s', '1-2m', '2-3m', '3-4m', '4-5m', '5-6m', '6-7m', '7m+']
        )

        # 2. Display size and range
        X['video_display_size'] = X['server_height'] * X['server_width']
        X['video_display_size_range'] = pd.cut(
            X['video_display_size'], 
            bins=[0, 1.2e+07, 2.4e+07, 3.6e+07, np.inf], 
            labels=['small', 'small-medium', 'medium-large', 'large']
        )

        # 3. Video age and range
        upload_dt = pd.to_datetime(X['upload_dt'], unit='s', errors='coerce')
        X['video_age'] = (self.max_upload_dt_ - upload_dt).dt.days
        X['video_age_range'] = '0-7d'
        X['video_age_range'] = pd.cut(
            X['video_age'], 
            bins=[-1, 7, 14, 30, 60, 90, np.inf], 
            labels=['0-7d', '7-14d', '14-30d', '30-60d', '60-90d', '90d+']
        )

        # 4. Author class
        X['author_class'] = 'smaller-creator'
        for group, creators in self.top_creators.items():
            X.loc[X['author_id'].isin(creators), 'author_class'] = group

        # 5. Song rank
        X['song_rank'] = 'other'
        for group, songs in self.top_songs.items():
            X.loc[X['music_id'].isin(songs), 'song_rank'] = group

        # 6. Extract first tag
        X['tag'] = X['tag'].apply(lambda x: (str(x).split(',') if pd.notna(x) else [-1])[0]).astype(str)

        # Rename using descriptions
        X = X.rename(columns=self.field_map)
        
        return X
    
    def fit_transform(self, X, y = None, **fit_params):
        return super().fit_transform(X, y, **fit_params)


# Configuration for the Pipeline
CATEGORICALS = [
  "Type of this video.",
  "The upload type of this video.",
  "The visible state of this video on the APP now.",
  "Background music type of this video.",
  "The duration range of this video.",
  "The display area range of this video.",
  "The age range of this video.",
  "The class of the author of this video.",
  "The rank of the background music of this video in the music library.",
  "Video hashtag.",
]

NUMERICALS = [
  "The time duration of this video (in millisecond).",
  "The display area of this video.",
  "The age of this video (in days).",
]

VideoFeaturePreprocessor = Pipeline(steps=[
    ('engineering', VideoFeatureEngineering()),
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), CATEGORICALS),
            ('num', 'passthrough', NUMERICALS)
        ]
    ))
])