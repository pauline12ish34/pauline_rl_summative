"""
Simple FastAPI app to serve agent results as JSON.
Run with: uvicorn api:app --reload
"""

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
import pandas as pd
import os

app = FastAPI()


# Serve *_experiments.csv and *_results.csv
@app.get("/experiments/{algo}")
def get_experiments(algo: str):
    path = f"results/{algo.lower()}_experiments.csv"
    if not os.path.exists(path):
        return {"error": "Experiments not found."}
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/results/{algo}")
def get_results(algo: str):
    path = f"results/{algo.lower()}_results.csv"
    if not os.path.exists(path):
        return {"error": "Results not found."}
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

# Serve plot images from results/plots/
@app.get("/plots/{filename}")
def get_plot(filename: str):
    plot_path = os.path.join("results", "plots", filename)
    if not os.path.exists(plot_path):
        return {"error": "Plot not found."}
    return FileResponse(plot_path)


@app.get("/")
def root():
    return {"message": "RL Agent Results API. Use /results/{algo}, /experiments/{algo}, /plots/{filename}"}
