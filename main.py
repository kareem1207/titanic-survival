from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
from pathlib import Path

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the model
model_path = Path(__file__).parent / "titanic_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    pclass: int = Form(...),
    age: float = Form(...),
    sibsp: int = Form(...),
    parch: int = Form(...),
    fare: float = Form(...),
    gender_male: int = Form(...),
    embarked_q: int = Form(...),
    embarked_s: int = Form(...)
):
    print(f"Received: pclass={pclass}, age={age}, sibsp={sibsp}, parch={parch}, fare={fare}, gender_male={gender_male}, embarked_q={embarked_q}, embarked_s={embarked_s}")
    
    # Create feature array: [Pclass, Age, SibSp, Parch, Fare, Gender_male, Embarked_Q, Embarked_S]
    features = np.array([[pclass, age, sibsp, parch, fare, gender_male, embarked_q, embarked_s]])
    
    print(f"Features: {features}")
    
    # Make prediction
    prediction_value = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    print(f"Prediction: {prediction_value}")
    
    # Create result dictionary matching template expectations
    result = {
        "survived": "Yes" if prediction_value == 1 else "No",
        "probability": f"{probability[prediction_value]*100:.2f}%"
    }
    
    return templates.TemplateResponse("index.html", {"request": request, "prediction": result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
