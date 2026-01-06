Video Link : https://drive.google.com/file/d/1JPSuABTXTKmNrfY3S9FD7sVbqzRtzne7/view?usp=sharing 

test the deployed Heart Disease Prediction API locally using Docker + Kubernetes (Helm) without setting up Python, virtual environments, or ML code.
________________________________________
Prerequisites (Only Once)
The examiner needs only the following installed:
1.	Docker Desktop (Windows / Mac / Linux)
o	Download: https://www.docker.com/products/docker-desktop
o	Enable Kubernetes from:
o	Docker Desktop → Settings → Kubernetes → Enable
2.	kubectl (usually bundled with Docker Desktop)
3.	Helm (optional – already used by the author)
No Python, pip, virtualenv, or ML libraries are required.
________________________________________
Step 1: Pull the Docker Image
docker pull shashankkarnatisai/mlops-assignment-api:v1
________________________________________
Step 2: Run the API Container
docker run -p 8000:8000 shashankkarnatisai/mlops-assignment-api:v1
✔ API starts automatically
✔ Model loads from the container
✔ Logs are printed to console
________________________________________
Step 3: Verify API is Running
Open browser:
http://localhost:8000/health
Expected response:
{
  "status": "UP",
  "model_loaded": true,
  "uptime_seconds": 12
}
________________________________________
Step 4: Test Prediction API
Endpoint
POST http://localhost:8000/predict
Sample JSON Input
{
  "age": 54,
  "sex": 1,
  "cp": 3,
  "trestbps": 130,
  "chol": 246,
  "fbs": 0,
  "restecg": 1,
  "thalach": 173,
  "exang": 0,
  "oldpeak": 0.0,
  "slope": 2,
  "ca": 0,
  "thal": 3
}
Sample Output
{
    "prediction": 0,
    "confidence": 0.097
}
________________________________________
Step 5: View Monitoring Metrics
Open:
http://localhost:8000/metrics
Shows:
•	Total requests
•	Prediction distribution
•	Latency
•	Model ROC-AUC
•	Uptime
