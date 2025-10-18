# 🐱 Are You a Cat? MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8+-0194E2.svg)](https://mlflow.org/)
[![Airflow](https://img.shields.io/badge/Airflow-2.7+-017CEE.svg)](https://airflow.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A complete end-to-end MLOps pipeline demonstrating Level 3 MLOps maturity with automated training, tracking, deployment, monitoring, and feedback loops.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Monitoring & Observability](#-monitoring--observability)
- [MLOps Maturity Level](#-mlops-maturity-level)
- [Learning Objectives](#-learning-objectives)
- [Development Roadmap](#-development-roadmap)
- [Contributing](#-contributing)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🎯 Overview

This project implements a **production-grade MLOps pipeline** for binary image classification (cat vs. not-cat). The primary goal is **not model accuracy**, but rather to showcase:

✅ **Automated ML pipelines** with experiment tracking  
✅ **Orchestrated workflows** with Apache Airflow  
✅ **Model registry and versioning** with MLflow  
✅ **Real-time model serving** via REST API  
✅ **User feedback collection** for continuous improvement  
✅ **Automated retraining triggers** based on performance metrics  
✅ **Comprehensive monitoring** with Prometheus & Grafana

**This project is designed for learning and portfolio demonstration of MLOps engineering skills.**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    Streamlit Web App                            │
│              (Upload Image → Get Prediction)                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL SERVING                              │
│                  MLflow Model Server                            │
│              (REST API → Production Model)                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EXPERIMENT TRACKING                           │
│                       MLflow                                    │
│        (Track Experiments, Register Models, Artifacts)          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PIPELINE ORCHESTRATION                         │
│                    Apache Airflow                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Training DAG │  │ Deploy DAG   │  │ Monitor DAG  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│               MONITORING & OBSERVABILITY                        │
│              Prometheus + Grafana                               │
│     (Metrics, Dashboards, Alerts, Drift Detection)             │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Training Pipeline** (Airflow DAG):

   - Load and preprocess data
   - Train CNN model
   - Log experiments to MLflow
   - Evaluate model performance
   - Promote to production if metrics pass threshold

2. **Deployment Pipeline** (Airflow DAG):

   - Serve production model via MLflow
   - Expose REST API endpoint

3. **User Interaction** (Streamlit):

   - Upload image
   - Get prediction from model server
   - Provide feedback (correct/incorrect)

4. **Monitoring Pipeline** (Airflow DAG):
   - Check model performance metrics
   - Analyze user feedback
   - Trigger retraining if performance degrades
   - Monitor data drift

---

## ✨ Features

### Core MLOps Capabilities

- **🔄 Automated Training Pipeline**

  - Scheduled model retraining
  - Hyperparameter tracking
  - Experiment versioning

- **📊 Experiment Tracking**

  - MLflow integration
  - Metric logging (accuracy, precision, recall, F1)
  - Artifact storage (models, plots, datasets)

- **🚀 Model Registry & Versioning**

  - Staged deployments (Staging → Production)
  - Model lineage tracking
  - A/B testing support

- **🌐 Model Serving**

  - REST API for predictions
  - Scalable inference service
  - Low-latency responses

- **💬 Feedback Loop**

  - User feedback collection
  - Ground truth labeling
  - Feedback-driven retraining

- **📈 Monitoring & Alerting**

  - Real-time performance tracking
  - Data drift detection
  - Automated retraining triggers
  - Custom Grafana dashboards

- **🐳 Containerization**
  - Docker-based deployment
  - Docker Compose orchestration
  - Easy reproducibility

---

## 🛠️ Technology Stack

| Layer                   | Technology              | Purpose                            |
| ----------------------- | ----------------------- | ---------------------------------- |
| **ML Framework**        | TensorFlow/Keras        | CNN model development              |
| **Experiment Tracking** | MLflow                  | Track experiments, register models |
| **Orchestration**       | Apache Airflow          | Automate workflows, scheduling     |
| **Model Serving**       | MLflow Serve            | REST API for predictions           |
| **UI**                  | Streamlit               | User interface for predictions     |
| **Monitoring**          | Prometheus              | Metrics collection                 |
| **Visualization**       | Grafana                 | Dashboards and alerts              |
| **Drift Detection**     | DeepChecks/Evidently    | Data quality monitoring            |
| **Containerization**    | Docker + Docker Compose | Service orchestration              |
| **Database**            | PostgreSQL              | Airflow metadata                   |
| **Storage**             | Local/S3                | Artifact storage                   |

---

## 📁 Project Structure

```
are-you-a-cat-mlops-pipeline/
│
├── dags/                          # Airflow DAGs
│   ├── train_pipeline.py         # Training orchestration
│   ├── deploy_pipeline.py        # Deployment automation
│   └── monitor_pipeline.py       # Monitoring and retraining triggers
│
├── src/                           # Core ML code
│   ├── data_loader.py            # Dataset loading utilities
│   ├── preprocess.py             # Image preprocessing
│   ├── model_train.py            # Model training logic
│   ├── evaluate.py               # Model evaluation
│   └── inference.py              # Prediction logic
│
├── app/                           # Streamlit application
│   └── streamlit_app.py          # User interface
│
├── notebooks/                     # Jupyter notebooks
│   └── exploratory_analysis.ipynb
│
├── data/                          # Data storage
│   ├── raw/                      # Raw dataset
│   ├── processed/                # Preprocessed data
│   └── feedback/                 # User feedback logs
│
├── models/                        # Saved models (if not using MLflow)
│
├── mlruns/                        # MLflow tracking data
│
├── config/                        # Configuration files
│   ├── prometheus.yml            # Prometheus config
│   └── grafana/                  # Grafana dashboards
│
├── tests/                         # Unit and integration tests
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
│
├── docker-compose.yml             # Multi-service orchestration
├── Dockerfile                     # Container definition
├── requirements.txt               # Python dependencies
├── .gitignore                    # Git ignore rules
├── .dockerignore                 # Docker ignore rules
├── README.md                      # Project documentation
└── LICENSE                        # MIT License
```

---

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

- **Docker** (>= 20.10)
- **Docker Compose** (>= 2.0)
- **Python** (>= 3.10) - for local development
- **Git**
- At least **8GB RAM** available for Docker
- **10GB disk space** for images and artifacts

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/bigalex95/are-you-a-cat-mlops-pipeline.git
cd are-you-a-cat-mlops-pipeline
```

### 2. Start All Services

```bash
docker-compose up -d
```

This will start:

- **MLflow Server** on `http://localhost:5000`
- **Airflow Webserver** on `http://localhost:8080`
- **Streamlit App** on `http://localhost:8501`
- **Prometheus** on `http://localhost:9090`
- **Grafana** on `http://localhost:3000`

### 3. Access the Services

| Service       | URL                   | Default Credentials |
| ------------- | --------------------- | ------------------- |
| MLflow UI     | http://localhost:5000 | No auth             |
| Airflow UI    | http://localhost:8080 | admin / admin       |
| Streamlit App | http://localhost:8501 | No auth             |
| Grafana       | http://localhost:3000 | admin / admin       |
| Prometheus    | http://localhost:9090 | No auth             |

### 4. Trigger Training Pipeline

1. Open Airflow UI at `http://localhost:8080`
2. Enable the `train_pipeline` DAG
3. Click "Trigger DAG" to start training
4. Monitor progress in the DAG graph view

### 5. Make Predictions

1. Open Streamlit app at `http://localhost:8501`
2. Upload a cat or non-cat image
3. View prediction results
4. Provide feedback (correct/incorrect)

---

## 📖 Usage Guide

### Training a Model

#### Option 1: Via Airflow (Automated)

```bash
# Trigger training DAG via CLI
docker exec -it airflow-webserver airflow dags trigger train_pipeline
```

#### Option 2: Local Training (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Run training script
python src/model_train.py
```

### Viewing Experiments

Navigate to MLflow UI and explore:

- **Experiments**: Compare runs with different hyperparameters
- **Models**: View registered models and their versions
- **Artifacts**: Download trained models, plots, datasets

### Deploying a Model

The deployment DAG automatically serves the latest production model:

```bash
# Check model serving status
curl http://localhost:5001/ping

# Make prediction via API
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"instances": [<image_data>]}'
```

### Monitoring

Access Grafana dashboards to view:

- Model accuracy over time
- Prediction latency
- Request rate and error rate
- Data drift metrics
- Feedback statistics

---

## 📊 Monitoring & Observability

### Key Metrics Tracked

1. **Model Performance**

   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix
   - ROC-AUC curve

2. **Service Health**

   - API response time
   - Request rate (predictions/sec)
   - Error rate
   - CPU/Memory usage

3. **Data Quality**

   - Input distribution shifts
   - Feature drift
   - Label distribution

4. **Business Metrics**
   - Feedback rate
   - User satisfaction
   - Prediction confidence distribution

### Automated Retraining

The monitoring DAG checks performance daily and triggers retraining if:

- Accuracy drops below threshold (e.g., 75%)
- Data drift is detected
- Sufficient new feedback data is collected

---

## 🎓 MLOps Maturity Level

This project demonstrates **Level 3 MLOps Maturity**:

| Level       | Description                     | This Project                      |
| ----------- | ------------------------------- | --------------------------------- |
| **Level 0** | Manual process                  | ❌                                |
| **Level 1** | ML pipeline automation          | ✅ Airflow DAGs                   |
| **Level 2** | CI/CD for ML                    | ✅ Automated deployment           |
| **Level 3** | Automated training + monitoring | ✅ Feedback loop, drift detection |
| **Level 4** | Full MLOps                      | 🔄 In progress                    |

### Level 3 Features Implemented

✅ Automated training pipelines  
✅ Experiment tracking and versioning  
✅ Model registry with staging  
✅ Automated deployment  
✅ Model monitoring  
✅ Data validation  
✅ Feedback loop  
✅ Drift detection  
✅ Automated retraining triggers

---

## 🎯 Learning Objectives

By building and exploring this project, you will learn:

### 1. MLOps Fundamentals

- End-to-end ML pipeline design
- Separation of concerns (data, training, serving, monitoring)
- Reproducibility with containers

### 2. Experiment Tracking

- Logging hyperparameters and metrics
- Comparing model runs
- Artifact management

### 3. Orchestration

- Building Airflow DAGs
- Task dependencies and XComs
- Scheduling and triggers

### 4. Model Serving

- REST API design for ML models
- Model versioning and rollback
- Scalable inference

### 5. Monitoring

- Metrics collection and visualization
- Alerting on performance degradation
- Data drift detection

### 6. DevOps for ML

- Docker containerization
- Multi-service orchestration
- Configuration management

---

## 🗺️ Development Roadmap

### ✅ Phase 1: Foundation

- [x] Project structure setup
- [x] Docker environment
- [x] Basic documentation

### 🚧 Phase 2: Core Pipeline (In Progress)

- [ ] Data loading and preprocessing
- [ ] CNN model training
- [ ] MLflow integration

### 📅 Phase 3: Automation

- [ ] Airflow training DAG
- [ ] Airflow deployment DAG
- [ ] Model registry integration

### 📅 Phase 4: User Interface

- [ ] Streamlit app development
- [ ] Feedback collection
- [ ] API integration

### 📅 Phase 5: Monitoring

- [ ] Prometheus metrics
- [ ] Grafana dashboards
- [ ] Drift detection

### 📅 Phase 6: Advanced Features

- [ ] A/B testing support
- [ ] Multi-model comparison
- [ ] Advanced drift detection
- [ ] CI/CD with GitHub Actions

### 📅 Phase 7: Production Readiness

- [ ] Kubernetes deployment
- [ ] Cloud integration (AWS/GCP/Azure)
- [ ] Security hardening
- [ ] Load testing

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests** (if applicable)
5. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Code Style

- Follow PEP 8 for Python code
- Add docstrings to all functions
- Include type hints
- Write unit tests for new features

### Areas for Contribution

- 🐛 Bug fixes
- ✨ New features (see Roadmap)
- 📝 Documentation improvements
- 🧪 Additional tests
- 🎨 UI/UX enhancements

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Docker Containers Won't Start

```bash
# Check Docker daemon is running
docker info

# Check logs
docker-compose logs <service-name>

# Rebuild containers
docker-compose down -v
docker-compose up --build
```

#### 2. Airflow Webserver Not Accessible

```bash
# Check Airflow initialization
docker exec -it airflow-webserver airflow db check

# Reinitialize database
docker exec -it airflow-webserver airflow db init
```

#### 3. MLflow Tracking Not Working

```bash
# Check MLflow server logs
docker-compose logs mlflow-server

# Verify tracking URI
echo $MLFLOW_TRACKING_URI
```

#### 4. Out of Memory Errors

```bash
# Increase Docker memory limit in Docker Desktop settings
# Recommended: At least 8GB RAM for Docker
```

### Getting Help

- 📖 Check the [Issues](https://github.com/bigalex95/are-you-a-cat-mlops-pipeline/issues) page
- 💬 Open a new issue with detailed error logs
- 📧 Contact: [Your Email]

---

## 📚 Resources & References

### Official Documentation

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)

### Learning Materials

- [Made With ML - MLOps](https://madewithml.com/)
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/)
- [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)

### Datasets

- [Cats vs Dogs Dataset](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- TensorFlow team for the excellent ML framework
- MLflow community for experiment tracking tools
- Apache Airflow for workflow orchestration
- Streamlit for easy UI development

---

## 📞 Contact

**Alibek Erkabayev** - [@bigalex95](https://github.com/bigalex95)

Project Link: [https://github.com/bigalex95/are-you-a-cat-mlops-pipeline](https://github.com/bigalex95/are-you-a-cat-mlops-pipeline)

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

---

<div align="center">

**Built with ❤️ for learning and demonstrating MLOps best practices**

</div>
