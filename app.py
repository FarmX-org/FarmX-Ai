from fastapi import FastAPI
from pydantic import BaseModel
from recommender import crop_ai
from utils.preprocess import extract_soil_tag

app = FastAPI()

class CropRequest(BaseModel):
    soil_type: str
    season: str

class CropRecommendation(BaseModel):
    recommended_crop: str

@app.post("/recommend", response_model=CropRecommendation)
async def recommend(request: CropRequest):
    soil_tag = extract_soil_tag(request.soil_type)
    season = request.season
    try:
        recommendation = crop_ai.predict(soil_tag, season)
        return {"recommended_crop": recommendation}
    except Exception as e:
        return {"recommended_crop": f"Error: {str(e)}"}

@app.get("/")
async def root():
    return {"message": "Welcome to FarmX Crop Recommender AI"}
