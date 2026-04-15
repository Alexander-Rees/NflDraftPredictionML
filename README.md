# NFL Draft Prediction 

**Alexander Rees · Andrew Lotocki**  
DS 4420

## Models

**1. Neural network (drafted or not)**  
A simple multi layer perceptron (MLP) classifier that predicts whether a player will be drafted within their position group. It takes historical combine and college data and outputs a prediction using a sigmoid activation.

**2. Bayesian model (draft position)**  
A Bayesian logistic regression model that predicts draft outcomes for individual players at a position level. It also takes in combine and college data and outputs a prediction for whether a player is drafted or not. 


## Data sources

- **CBS Sports** — 2025 NFL Scouting Combine invite list (329 prospects).
- **College Football Data API** — college stats and context.
- **Kaggle (Thomas Shaw)** — NFL Combine performance dataset.
