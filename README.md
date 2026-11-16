# are-you-a-cat-mlops-pipeline

Learning project: a cats-vs-dogs image classifier with a data preprocessing module, a small Keras CNN, and DVC data versioning. The MLOps parts around it (orchestration, serving, monitoring) are planned, not built. Work in progress, not maintained.

## What exists

- `src/data_loader.py` - downloads/loads Cats vs Dogs via TensorFlow Datasets (with caching).
- `src/preprocess.py` - resize/normalize, augmentation generator, train/val/test split, save/load processed arrays.
- `src/model_train.py` - builds a 3-block CNN (150x150x3 input, sigmoid output), compiles, trains with callbacks (early stopping, checkpointing, LR reduction), saves/loads models.
- `notebooks/` - Colab/local notebooks for preprocessing, training and a real-time classifier demo.
- DVC setup (`.dvc/`, `data/*.dvc`, `models.dvc`, `scripts/*.sh`) with DagsHub and Backblaze B2 remotes; see [docs/DVC_SETUP.md](docs/DVC_SETUP.md).
- `Dockerfile` and `docker-compose.yml`: MLflow server, Streamlit, and Airflow/Postgres service definitions (scaffold, see Status).
- `tests/test_data_pipeline.py`, `tests/test_model_with_real_data.py` - integration scripts for the data pipeline and training.

## Structure

```
src/        data_loader.py, preprocess.py, model_train.py
notebooks/  5 notebooks
tests/      2 integration test scripts
scripts/    DVC helper scripts
docs/       DVC setup, memory/GPU notes, test report, model notes
app/        streamlit_app.py (empty placeholder)
dags/       empty (mounted by docker-compose for Airflow)
```

## Run

```bash
pip install -r requirements-local.txt
python src/data_loader.py          # downloads the dataset via TFDS and loads a sample
python tests/test_data_pipeline.py
```

I did not re-run these for this cleanup. Pulling the DVC-tracked data needs access to the configured remotes.

## Results

Only a demo run is documented, in [docs/MODEL_CAPABILITIES.md](docs/MODEL_CAPABILITIES.md): 1,000 images, 5 epochs, test accuracy about 51%, validation about 55%. This is a smoke test, not a trained model, and there is no full-training evaluation.

## Status and roadmap

| Item | State |
|---|---|
| Data loading, preprocessing, CNN training | Implemented |
| DVC data/model versioning | Configured |
| MLflow tracking in code | Compose service only; not wired into the training code |
| Streamlit app | Placeholder file, compose service refers to it |
| Evaluation, inference | Planned |
| Airflow DAGs (train/deploy/monitor) | Planned; compose services exist, no DAGs |
| Prometheus/Grafana monitoring | Planned |
| Data validation (e.g. DeepChecks), ZenML orchestration, CI | Not started; earlier versions of this README wrongly described them as done |
| Unit tests (preprocessing, model, API) | Planned |

## Credits

Dataset: Cats vs Dogs via TensorFlow Datasets. Built with TensorFlow/Keras, DVC and MLflow. Licensed MIT, see `LICENSE`.
