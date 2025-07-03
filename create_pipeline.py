import joblib
from sklearn.pipeline import Pipeline

print("Loading the scaler and k-means model...")

try:
    # Load the individual components
    scaler = joblib.load("scaler.pkl")
    kmeans_model = joblib.load("kmeans_model.pkl")

    # Create a scikit-learn Pipeline
    # This chains the scaler and the model together. When you call .predict() on the pipeline,
    # it will automatically scale the data first and then pass it to the k-means model.
    pipeline = Pipeline([
        ('scaler', scaler),
        ('kmeans', kmeans_model)
    ])

    # Save the entire pipeline to a single file
    joblib.dump(pipeline, "pipeline.pkl")

    print("\nSUCCESS: 'pipeline.pkl' created successfully.")
    print("This single file now contains both your scaler and your model.")
    print("You are ready for the next step: uploading this file to Hugging Face Hub.")

except FileNotFoundError:
    print("\nERROR: Could not find 'scaler.pkl' or 'kmeans_model.pkl'.")
    print("Please make sure this script is in the same folder as your model files.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")