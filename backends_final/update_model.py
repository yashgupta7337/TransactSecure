import xgboost as xgb

# Define the path to your old model
old_model_path = "/Users/yashgupta/Desktop/BerTrugS-main/backends_final/final_model_5.bin"
# Define the path for the new model
new_model_path = "/Users/yashgupta/Desktop/BerTrugS-main/backends_final/updated_final_model_5.bin"

# Load the old model
old_model = xgb.Booster()
old_model.load_model(old_model_path)

# Save the model again with the updated format
old_model.save_model(new_model_path)
print(f"Model has been updated and saved to {new_model_path}")
