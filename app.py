from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Malaria Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load model artifacts at startup
try:
    # Load model artifacts at startup
    model = joblib.load('xgb_boost_model.joblib')
    scaler = joblib.load('scaler.joblib')
    with open('encoding_mappings.pkl', 'rb') as f:
        encoding_mappings = pickle.load(f)
except Exception as e:
    logger.error(f"Failed to load model artifacts: {str(e)}")
    raise Exception(f"Failed to load model artifacts: {str(e)}")

# Pydantic model for input data
class PredictionInput(BaseModel):
    Year: int
    Epidemic_Week: int
    Month: str
    RegionName: Optional[str] = None
    ZoneName: Optional[str] = None
    WoredaName: Optional[str] = None
    TMSuspected_Fever_Examined: Optional[float] = None
    temp_max: Optional[float] = None
    temp_min: Optional[float] = None
    temp_mean: Optional[float] = None
    rainfall: Optional[float] = None
    rel_humidity_mean: Optional[float] = None
    rel_humidity_max: Optional[float] = None
    rel_humidity_min: Optional[float] = None

class PredictionRequest(BaseModel):
    data: List[PredictionInput]

def data_preprocessing(df, scaler, encoding_mappings):
    """
    Preprocess input data for inference.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        scaler (StandardScaler): Loaded scaler.
        encoding_mappings (dict): Loaded encoding mappings.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    numerical_cols = ['TMSuspected Fever Examined', 'temp_max', 'temp_min', 'temp_mean', 
                      'rainfall', 'rel_humidity_mean', 'rel_humidity_max', 'rel_humidity_min']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = ['RegionName', 'ZoneName', 'WoredaName']
    for col in categorical_cols:
        if col in df.columns and encoding_mappings[col] is not None:
            df[col + '_encoded'] = df[col].map(encoding_mappings[col])
            df[col + '_encoded'] = df[col + '_encoded'].fillna(encoding_mappings[col].mean())
        else:
            df[col + '_encoded'] = encoding_mappings[col].mean() if encoding_mappings[col] is not None else 0

    cols_to_drop = [col for col in categorical_cols if col in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop, axis=1)

    if any(col in df.columns for col in numerical_cols):
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    logger.info("Data preprocessing completed")
    return df

def feature_engineering(df):
    """
    Engineer features for inference, ensuring all season columns are present.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    required_engineered_cols = ['Month_sin', 'Month_cos', 'season_Bega', 'season_Belg', 'season_Kiremt', 'season_Tseday']
    is_engineered = set(required_engineered_cols).issubset(df.columns)

    if not is_engineered:
        if 'Month' in df.columns:
            month_mapping = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
                'july ': 7, 'septamber': 9, 'january ': 1, 'february ': 2, 'feburary': 2,
                'april?': 4, 'may ': 5, 'june ': 6
            }
            
            df['Month_cleaned'] = df['Month'].str.strip().str.replace('?', '', regex=False).str.lower()
            df['Month_numeric'] = df['Month_cleaned'].map(month_mapping)
            
            if df['Month_numeric'].isna().any():
                unmapped = df[df['Month_numeric'].isna()]['Month_cleaned'].unique()
                logger.error(f"Unmapped Month values: {unmapped}")
                raise ValueError(f"Unmapped 'Month' values: {unmapped}. Please provide valid month names.")

            def map_season(month):
                if month in [6, 7, 8]:
                    return 'Kiremt'
                elif month in [9, 10, 11]:
                    return 'Tseday'
                elif month in [12, 1, 2]:
                    return 'Bega'
                elif month in [3, 4, 5]:
                    return 'Belg'
                else:
                    return 'Unknown'

            df['season'] = df['Month_numeric'].apply(map_season)
            
            # Initialize all season columns with zeros
            for season in ['season_Bega', 'season_Belg', 'season_Kiremt', 'season_Tseday']:
                df[season] = 0
            
            # Set the appropriate season column to 1
            for idx, season in df['season'].items():
                if season != 'Unknown':
                    df.at[idx, f'season_{season}'] = 1

            df['Epidemic_Week_sin'] = np.sin(2 * np.pi * df['Epidemic_Week'] / 53)
            df['Epidemic_Week_cos'] = np.cos(2 * np.pi * df['Epidemic_Week'] / 53)
            df['Month_sin'] = np.sin(2 * np.pi * df['Month_numeric'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month_numeric'] / 12)

            df = df.drop(['Month', 'Month_cleaned', 'Month_numeric', 'season'], axis=1)
        else:
            logger.error("Month column missing and engineered features not found")
            raise KeyError("Month column missing and engineered features (Month_sin, Month_cos, season_*) not found in input data.")

    if 'Year' in df.columns and 'Epidemic_Week' in df.columns:
        min_year = model.min_year if hasattr(model, 'min_year') else 2015  # Default min_year
        df['Year_Diff'] = df['Year'] - min_year
        df['Time_Index'] = (df['Year'] - min_year) * 53 + df['Epidemic_Week']
    else:
        df['Year_Diff'] = 0
        df['Time_Index'] = 0

    df['Malaria_Lag1'] = 0
    df['Malaria_Lag2'] = 0
    df['Temp_Rainfall'] = df['temp_mean'] * df['rainfall'] if 'temp_mean' in df.columns and 'rainfall' in df.columns else 0
    df['Temp_Humidity'] = df['temp_mean'] * df['rel_humidity_mean'] if 'temp_mean' in df.columns and 'rel_humidity_mean' in df.columns else 0
    df['Malaria_Rolling_Mean'] = 0

    logger.info("Feature engineering completed")
    return df

def model_prediction(model, scaler, encoding_mappings, new_data):
    """
    Predict malaria cases for new data.
    
    Args:
        model (xgb.XGBRegressor): Trained model.
        scaler (StandardScaler): Scaler for numerical features.
        encoding_mappings (dict): Mappings for target encoding.
        new_data (pd.DataFrame): New data for prediction.
    
    Returns:
        np.array: Predicted malaria cases.
    """
    try:
        new_data = data_preprocessing(new_data, scaler, encoding_mappings)
        new_data = feature_engineering(new_data)
        
        features = ['Epidemic_Week', 'TMSuspected Fever Examined', 
                    'temp_max', 'temp_min', 'temp_mean', 'rainfall', 
                    'rel_humidity_mean', 'rel_humidity_max', 'rel_humidity_min',
                    'RegionName_encoded', 'ZoneName_encoded', 'WoredaName_encoded',
                    'Epidemic_Week_sin', 'Epidemic_Week_cos', 'Month_sin', 'Month_cos',
                    'season_Bega', 'season_Belg', 'season_Kiremt', 'season_Tseday',
                    'Malaria_Lag1', 'Malaria_Lag2', 'Temp_Rainfall', 'Temp_Humidity',
                    'Malaria_Rolling_Mean', 'Year_Diff', 'Time_Index']
        
        # Ensure all features are present
        for feature in features:
            if feature not in new_data.columns:
                new_data[feature] = 0
                logger.warning(f"Feature {feature} missing, filled with 0")
        
        new_data = new_data[features]
        
        predictions = model.predict(new_data)
        logger.info("Predictions generated successfully")
        return predictions
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

@app.post("/predict", response_model=dict)
async def predict(request: PredictionRequest):
    """
    Predict malaria cases for input data.
    
    Args:
        request (PredictionRequest): Input data as a list of PredictionInput.
    
    Returns:
        dict: Predicted malaria cases.
    """
    try:
        input_data = [item.dict() for item in request.data]
        df = pd.DataFrame(input_data)
        
        df = df.rename(columns={'TMSuspected_Fever_Examined': 'TMSuspected Fever Examined'})
        
        predictions = model_prediction(model, scaler, encoding_mappings, df)
        # predictions = predictions.tolist()
        # print("predictions", predictions)
        # return {"number_of_malaria_cases": round(predictions[0])}
        return {"predictions": predictions.tolist()}
    
    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except KeyError as ke:
        logger.error(f"KeyError: {str(ke)}")
        raise HTTPException(status_code=400, detail=str(ke))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    """
    Root endpoint for health check.
    
    Returns:
        dict: Welcome message.
    """
    return {"message": "Welcome to the Malaria Prediction API"}