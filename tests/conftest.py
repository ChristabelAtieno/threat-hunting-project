import pytest
from dotenv import load_dotenv

# live tests read AWS credentials from .env like the CLI does
load_dotenv()


@pytest.fixture(autouse=True)
def isolated_mlflow(request, tmp_path, monkeypatch):
    """Offline tests log to a throwaway MLflow store, never the real one holding your models."""
    if request.node.get_closest_marker('live'):
        return
    import mlflow
    monkeypatch.setenv('MLFLOW_TRACKING_URI', f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    yield
    mlflow.set_tracking_uri(None)
