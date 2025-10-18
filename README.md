# 🐱 Are You a Cat? – A Production-Grade MLOps System

![demo](assets/demo.gif)
_(Example: replace this with a short 5–10 sec GIF of your Streamlit app in action)_

---

## 🧠 Description

**Are You a Cat?** is not just another image classifier — it’s a **Level 3 MLOps system** that demonstrates what real production ML infrastructure looks like.

The goal:

> “Sometimes it’s hard to know if you are a cat or not. Upload a selfie and I’ll help you figure that out.”

While the model itself is a simple 2D CNN, the **real challenge** lies in the **MLOps automation**, **pipeline orchestration**, and **feedback loop**.
This project demonstrates **end-to-end ML system design**, including:

- 🚀 Automated pipelines (training → evaluation → deployment → inference)
- 🧩 Reproducibility & experiment tracking (MLflow)
- 🔍 Data validation (DeepChecks)
- 🤖 Automatic deployment on performance thresholds
- 🔄 Continuous learning feedback loop via Streamlit + S3

---

## 🧩 System Pipeline

![pipeline](assets/pipeline.png)
_(Example: include an architecture diagram showing ZenML orchestrating training, validation, MLflow deployment, Streamlit interface, and S3 feedback storage.)_

### 🔧 Pipeline Overview

| Stage                       | Tool           | Description                                           |
| :-------------------------- | :------------- | :---------------------------------------------------- |
| **1. Data Ingestion**       | ZenML          | Load cat/dog/selfie datasets from public sources      |
| **2. Data Validation**      | DeepChecks     | Detect data quality or distribution issues            |
| **3. Model Training**       | TensorFlow     | Train a CNN classifier (accuracy not the focus)       |
| **4. Experiment Tracking**  | MLflow         | Log parameters, metrics, and artifacts                |
| **5. Model Evaluation**     | ZenML + MLflow | Evaluate model using precision/recall thresholds      |
| **6. Auto Deployment**      | MLflow Serving | Deploy new model only if thresholds are met           |
| **7. Inference & Feedback** | Streamlit + S3 | Serve predictions and collect feedback for retraining |

---

## 📊 Results

Here’s an example of the deployed model in action:

| Input Image                   | Prediction   | Confidence |
| :---------------------------- | :----------- | :--------: |
| ![cat](assets/cat_sample.jpg) | 🐱 Cat       |    0.97    |
| ![dog](assets/dog_sample.jpg) | 🐶 Not a Cat |    0.02    |

> The system achieves **consistent reproducibility**, full **experiment logging**, and automated **model deployment** — the three pillars of Level 3 MLOps.

---

## 🏗️ Repository Structure

```bash
Are-You-A-Cat-MLOps-Pipeline/
│
├── pipelines/
│   ├── training_pipeline.py
│   ├── deployment_pipeline.py
│   └── inference_pipeline.py
│
├── src/
│   ├── data/
│   ├── models/
│   ├── utils/
│   └── validation/
│
├── app/
│   └── streamlit_app.py
│
├── configs/
│   ├── zenml_config.yaml
│   ├── mlflow_config.yaml
│   └── hyperparams.yaml
│
├── assets/
│   ├── demo.gif
│   ├── pipeline.png
│   └── cat_sample.jpg
│
├── requirements.txt
├── Dockerfile
├── README.md
└── LICENSE
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/Are-You-A-Cat-MLOps-Pipeline.git
cd Are-You-A-Cat-MLOps-Pipeline
```

### 2️⃣ Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Initialize ZenML and MLflow

```bash
zenml init
zenml integration install mlflow tensorflow deepchecks
mlflow ui
```

### 5️⃣ Configure ZenML Stack

```bash
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml stack register cat_stack \
    -a default \
    -o default \
    -d default \
    -e mlflow_tracker \
    --set
```

### 6️⃣ Run the pipelines

```bash
python pipelines/training_pipeline.py
python pipelines/deployment_pipeline.py
python pipelines/inference_pipeline.py
```

### 7️⃣ Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

### 8️⃣ View results

- 🧭 MLflow UI → [http://localhost:5000](http://localhost:5000)
- 🖥️ Streamlit App → [http://localhost:8501](http://localhost:8501)
- 📦 Stored data → S3 bucket or configured storage backend

---

## ☁️ Docker Setup (Optional)

To containerize the system:

```bash
docker build -t are-you-a-cat .
docker run -p 8501:8501 -p 5000:5000 are-you-a-cat
```

---

## 🔮 Future Improvements

- [ ] Integrate drift detection and alerting
- [ ] Move serving to Kubernetes (ZenML + KServe)
- [ ] Add CI/CD using GitHub Actions
- [ ] Implement automatic retraining from feedback data

---

## 📚 References

- [ZenML Documentation](https://docs.zenml.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DeepChecks Validation](https://docs.deepchecks.com/)
- [Streamlit](https://streamlit.io/)

---

## 🧑‍💻 Author

**Alibek Erkabayew**
Computer Vision & Machine Learning Engineer
🚀 Focused on scalable, production-ready MLOps systems.
