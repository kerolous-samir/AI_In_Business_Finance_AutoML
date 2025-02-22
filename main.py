from src.preprocess import preprocess_data
from src.train import train_xgboost

def main():
    print("Starting AI in Business (Finance) AutoML Project...
")
    
    print("Preprocessing data...")
    preprocess_data("data/1.1_UCI_Credit_Card.csv")
    
    print("Training model...")
    train_xgboost()

    print("Deployment step (if needed)...")
    # Uncomment below line if deployment is required
    # from src.deploy import deploy_model
    # deploy_model()

    print("Project completed!")

if __name__ == "__main__":
    main()
