import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import pytest


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir):
    """Provide a temporary file for tests."""
    temp_file_path = temp_dir / "test_file.txt"
    temp_file_path.write_text("test content")
    yield temp_file_path


@pytest.fixture
def mock_config():
    """Provide a mock configuration object."""
    config = MagicMock()
    config.learning_rate = 1e-4
    config.batch_size = 4
    config.num_epochs = 10
    config.model_name = "test_model"
    config.output_dir = "/tmp/test_output"
    return config


@pytest.fixture
def sample_image_path(temp_dir):
    """Provide a path for a sample test image."""
    return temp_dir / "sample_image.jpg"


@pytest.fixture
def sample_video_path(temp_dir):
    """Provide a path for a sample test video."""
    return temp_dir / "sample_video.mp4"


@pytest.fixture
def mock_torch_device():
    """Mock torch device for testing."""
    return MagicMock()


@pytest.fixture
def mock_model():
    """Provide a mock model for testing."""
    model = MagicMock()
    model.eval.return_value = model
    model.train.return_value = model
    model.parameters.return_value = []
    return model


@pytest.fixture
def mock_dataset():
    """Provide a mock dataset for testing."""
    dataset = MagicMock()
    dataset.__len__.return_value = 100
    dataset.__getitem__.return_value = {
        "image": MagicMock(),
        "text": "test caption",
        "label": 0
    }
    return dataset


@pytest.fixture
def mock_dataloader(mock_dataset):
    """Provide a mock dataloader for testing."""
    dataloader = MagicMock()
    dataloader.__iter__.return_value = iter([{
        "image": MagicMock(),
        "text": ["test caption"] * 4,
        "label": [0] * 4
    }])
    dataloader.__len__.return_value = 25
    return dataloader


@pytest.fixture
def sample_lora_config():
    """Provide a sample LoRA configuration."""
    return {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.1,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"]
    }


@pytest.fixture
def mock_text_encoder():
    """Provide a mock text encoder."""
    encoder = MagicMock()
    encoder.encode.return_value = MagicMock()
    return encoder


@pytest.fixture
def mock_tokenizer():
    """Provide a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "test text"
    return tokenizer


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test-specific environment variables
    os.environ["TESTING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture logs during tests."""
    yield caplog


class MockHuggingFaceModel:
    """Mock HuggingFace model for testing."""
    
    def __init__(self):
        self.config = MagicMock()
        self.state_dict = lambda: {}
    
    def eval(self):
        return self
    
    def train(self):
        return self
    
    def to(self, device):
        return self
    
    def __call__(self, *args, **kwargs):
        return MagicMock()


@pytest.fixture
def mock_huggingface_model():
    """Provide a mock HuggingFace model."""
    return MockHuggingFaceModel()