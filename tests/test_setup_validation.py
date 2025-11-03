"""
Validation tests to ensure the testing infrastructure is working correctly.
"""
import pytest
import sys
from pathlib import Path


def test_python_version():
    """Test that we're running a supported Python version."""
    assert sys.version_info >= (3, 10), "Python 3.10+ is required"
    assert sys.version_info < (3, 13), "Python version should be < 3.13"


def test_project_structure():
    """Test that the project structure is as expected."""
    project_root = Path(__file__).parent.parent
    
    # Check that main source directory exists
    src_dir = project_root / "src" / "musubi_tuner"
    assert src_dir.exists(), "Source directory should exist"
    assert (src_dir / "__init__.py").exists(), "Source package should be importable"
    
    # Check that pyproject.toml exists
    assert (project_root / "pyproject.toml").exists(), "pyproject.toml should exist"


def test_import_main_package():
    """Test that the main package can be imported."""
    try:
        import musubi_tuner
        assert hasattr(musubi_tuner, "__init__"), "Package should be importable"
    except ImportError:
        # This is expected if the package isn't installed yet
        pytest.skip("Package not installed, skipping import test")


def test_fixtures_available(temp_dir, mock_config, mock_model):
    """Test that our custom fixtures are working."""
    # Test temp_dir fixture
    assert temp_dir.exists(), "Temporary directory should exist"
    assert temp_dir.is_dir(), "temp_dir should be a directory"
    
    # Test mock_config fixture
    assert hasattr(mock_config, "learning_rate"), "Mock config should have learning_rate"
    assert mock_config.learning_rate == 1e-4, "Mock config should have expected values"
    
    # Test mock_model fixture
    assert hasattr(mock_model, "eval"), "Mock model should have eval method"
    assert hasattr(mock_model, "train"), "Mock model should have train method"


@pytest.mark.unit
def test_unit_marker():
    """Test that unit marker is working."""
    assert True, "Unit marker test should pass"


@pytest.mark.integration
def test_integration_marker():
    """Test that integration marker is working."""
    assert True, "Integration marker test should pass"


@pytest.mark.slow
def test_slow_marker():
    """Test that slow marker is working."""
    assert True, "Slow marker test should pass"


class TestClassNaming:
    """Test that test class naming conventions work."""
    
    def test_method_in_class(self):
        """Test method inside a test class."""
        assert True, "Test method in class should work"


def test_pytest_plugins():
    """Test that pytest plugins are available."""
    # Test pytest-mock
    try:
        import pytest_mock
        assert pytest_mock is not None
    except ImportError:
        pytest.fail("pytest-mock plugin is not available")
    
    # Test pytest-cov
    try:
        import pytest_cov
        assert pytest_cov is not None
    except ImportError:
        pytest.fail("pytest-cov plugin is not available")


def test_coverage_config():
    """Test that coverage configuration is working."""
    import coverage
    cov = coverage.Coverage()
    config = cov.get_option("run:source")
    
    # Check if our source directory is configured for coverage
    assert config is not None, "Coverage source should be configured"


def test_mock_functionality(mock_config, mocker):
    """Test that mocking functionality works correctly."""
    # Test that fixtures work
    assert mock_config.batch_size == 4
    
    # Test that pytest-mock is working
    mock_function = mocker.Mock(return_value=42)
    result = mock_function()
    assert result == 42
    mock_function.assert_called_once()


def test_temp_file_fixture(temp_file):
    """Test that temp_file fixture works correctly."""
    assert temp_file.exists(), "Temporary file should exist"
    assert temp_file.read_text() == "test content", "Temporary file should have expected content"


def test_environment_setup():
    """Test that test environment is set up correctly."""
    import os
    assert os.environ.get("TESTING") == "1", "TESTING environment variable should be set"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA should be disabled for tests"