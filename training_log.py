import wandb
import pandas as pd

api = wandb.Api()
run = api.run("dongheon97/RoadSafeAI-MS3D/812gee0n")

history_df = run.history(samples=10000) 

history_df.to_csv("wandb_training_log.csv", index=False)
print("Save as wandb_training_log.csv.")
