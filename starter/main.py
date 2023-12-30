from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI()

# Load model and encoders
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/lb.pkl")

class InferenceInput(BaseModel):
    age: int = Field(..., examples=[39])
    workclass: str = Field(..., alias="workclass", examples=["State-gov"])
    fnlgt: int = Field(..., examples=[77516])
    education: str = Field(..., alias="education", examples=["Bachelors"])
    education_num: int = Field(..., alias="education-num", examples=[13])
    marital_status: str = Field(..., alias="marital-status", examples=["Never-married"])
    occupation: str = Field(..., alias="occupation", examples=["Adm-clerical"])
    relationship: str = Field(..., alias="relationship", examples=["Not-in-family"])
    race: str = Field(..., examples=["White"])
    sex: str = Field(..., examples=["Male"])
    capital_gain: int = Field(..., alias="capital-gain", examples=[2174])
    capital_loss: int = Field(..., alias="capital-loss", examples=[0])
    hours_per_week: int = Field(..., alias="hours-per-week", examples=[40])
    native_country: str = Field(..., alias="native-country", examples=["United-States"]) 

@app.get("/")
async def root():
    return {"message": "Welcome to the ML model API!"}

@app.post("/predict")
async def predict(input_data: InferenceInput):
    try:
        # Process input data
        data_dict = input_data.dict(by_alias=True)
        df = pd.DataFrame([data_dict])
        X, _, _, _ = process_data(
            df,
            categorical_features=[
                "workclass",
                "education",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native-country",
            ],
            label=None,  # No label needed for inference
            training=False,  # Indicate it's not for training
            encoder=encoder,  # Use loaded encoder
            lb=lb,  # Use loaded label binarizer
        )

        # Make predictions
        preds = inference(model, X)

        # Assuming binary classification, convert prediction to label
        pred_label = lb.inverse_transform(preds)[0]

        return {"prediction": pred_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
