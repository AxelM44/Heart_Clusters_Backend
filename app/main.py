from typing import Union
from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from json_to_numpyUSA import *
from predict_diseaseUSA import *
from transfocentroidsUSA import *
from json_to_numpyLISS import *
from predict_diseaseLISS import *
from transfocentroidsLISS import *
import os
import json
import numpy as np

######## load settings from .env check dotenv online before app start

app = FastAPI()

@app.get("/healthcheck")
async def healthcheck():
    return {"is_alive": True}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

load_dotenv()
port = os.getenv('PORT')

# Create a router
api_router = APIRouter(prefix="/api/v1")

# Define the root endpoint
@api_router.get("/")
def read_root():
    return {"Hello": "World"}

json_file_path="app/values.json"
def load_values(file_path=json_file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

values = load_values()

# Define a POST endpoint to receive and return JSON data
@api_router.post("/predictionalgo1")
async def receive_json(request: Request):
    try:
        data = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    if len(data.keys()) == values["nb_question_USA"] :
        #transform the json file into a numpy array usable by kmeans
        transfo_dataUSA=json_to_numpyUSA(data)

        # Determine which cluster the patient is in and the caracteristics
        centroid_cl, analysis, index_list_means, index_list_perc =analyze_individual_dataUSA(transfo_dataUSA)

        # Transforming the centroid back to the original format and combining in a specific order the elements of two arrays : one contain means of the float values of a columns, the other the percentage of yes/no columns
        means_centroid, perc_centroid=invtransfo_centroidUSA(centroid_cl)

        # Combining all the data 
        combined_data = {
        "means_dataset": meansUSA,
        "perc_dataset": values["total_disease_percentUSA"]+tot_percent_columnUSA, # Adding the info that 8.53% of the dataset used for the clustering ahs heart disease
        "means_centroid": means_centroid,
        "perc_centroid": perc_centroid,
        "analysis": analysis,
        "index_list_means": index_list_means,
        "index_list_perc": index_list_perc,
        "Id": 'USA',
        }

        # Transforming the data into json
        data2return = json.dumps(combined_data)

    elif len(data.keys()) == values["nb_question_LISS"] :
        #transform the json file into a numpy array usable by kmeans
        transfo_dataLISS=json_to_numpyLISS(data)

        # Determine which cluster the patient is in and the caracteristics
        centroid_cl, analysis, index_list_means, index_list_perc = analyze_individual_dataLISS(transfo_dataLISS)

        # Transforming the centroid back to the original format and combining in a specific order the elements of two arrays : one contain means of the float values of a columns, the other the percentage of yes/no columns
        means_centroid, perc_centroid=invtransfo_centroidLISS(centroid_cl)

        means_dataset=invtransfo_norm_meansLISS(norm_meansLISS)

        combined_data ={
        "means_dataset": means_dataset,
        "perc_dataset": percLISS,
        "means_centroid": means_centroid,
        "perc_centroid": perc_centroid,
        "analysis": analysis,
        "index_list_means": index_list_means,
        "index_list_perc": index_list_perc,
        "Id": 'LISS',    
        }
        data2return = json.dumps(combined_data)
    
    return JSONResponse(content=data2return)

# Include the router in the FastAPI app
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
