import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from utils.preprocess import extract_soil_tag
from utils.db import get_connection

class CropRecommender:
    def __init__(self):
        self.soil_encoder = LabelEncoder()
        self.season_encoder = LabelEncoder()
        self.model = DecisionTreeClassifier()

    def load_data(self):
        conn = get_connection()
        query = "SELECT name, preferred_soil_type, season FROM crops"
        df = pd.read_sql(query, conn)
        conn.close()

        df['soil_tag'] = df['preferred_soil_type'].apply(extract_soil_tag)
        df['season'] = df['season'].str.lower()  # حوّل كل موسم لصغير

        soil_encoded = self.soil_encoder.fit_transform(df['soil_tag'])
        season_encoded = self.season_encoder.fit_transform(df['season'])

        X = list(zip(soil_encoded, season_encoded))
        y = df['name']

        return X, y

    def train(self):
        X, y = self.load_data()
        self.model.fit(X, y)
        print("Model trained and ready!")

    def predict(self, soil_input, season_input):
        soil_tag = extract_soil_tag(soil_input)
        season_input = season_input.lower()  # تأكد إن الادخال صغير

        soil_enc = self.soil_encoder.transform([soil_tag])[0]

        if season_input not in self.season_encoder.classes_:
            raise ValueError(f"Season '{season_input}' not recognized!")

        season_enc = self.season_encoder.transform([season_input])[0]
        pred = self.model.predict([[soil_enc, season_enc]])
        return pred[0]

crop_ai = CropRecommender()
crop_ai.train()
