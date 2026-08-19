# Development Guide for MassNet-DDA

This document outlines the development practices, testing procedures, and continuous integration setup for the MassNet-DDA project.

## Testing Framework

### Prerequisites

Before running tests, ensure you have the required dependencies:

```bash
pip install pytest pytest-cov flake8 black isort mypy
```

### Running Tests

We use pytest for our test suite. Here are common test commands:

```bash
# Run all tests with coverage report
pytest tests/ --cov=XuanjiNovo

# Run specific test file
pytest tests/test_model.py

# Run tests matching specific pattern
pytest -k "test_model"

# Run tests with detailed output
pytest -v tests/

# Run tests and generate HTML coverage report
pytest tests/ --cov=XuanjiNovo --cov-report=html
```

### Test Structure

- `tests/test_model.py`: Core model component tests
- `tests/conftest.py`: Shared pytest fixtures and configurations

## Code Quality Tools

We maintain code quality through several automated tools:

### Flake8 (Style Guide Enforcement)
```bash
# Check style
flake8 XuanjiNovo tests

# Common configuration in setup.cfg:
# [flake8]
# max-line-length = 100
# exclude = .git,__pycache__,build,dist
```

### Black (Code Formatting)
```bash
# Check formatting
black --check XuanjiNovo tests

# Apply formatting
black XuanjiNovo tests
```

### isort (Import Sorting)
```bash
# Check import sorting
isort --check-only XuanjiNovo tests

# Apply import sorting
isort XuanjiNovo tests
```

### mypy (Static Type Checking)
```bash
# Run type checking
mypy XuanjiNovo
```

## Continuous Integration

We use GitHub Actions for automated testing and quality assurance. The CI pipeline runs on:
- Push to main/master branches
- Pull request to main/master branches

### CI Pipeline Components

1. **Test Job**
   - Runs on Ubuntu with Python 3.8 and 3.9
   - Tests with CUDA 11.8.0
   - Generates coverage reports
   - Uploads coverage to Codecov

2. **Lint Job**
   - Runs all code quality tools
   - Ensures consistent code style
   - Checks type annotations

3. **Build Extensions Job**
   - Builds C++ extensions
   - Verifies CUDA compatibility
   - Tests compiled components

### CI Configuration

The complete CI configuration is in `.github/workflows/ci.yml`. Key features:
- Matrix testing across Python versions
- CUDA toolkit installation
- Dependency caching
- Automated coverage reporting

## Development Workflow

1. **Setting Up Development Environment**
   ```bash
   # Clone repository
   git clone https://github.com/your-username/MassNet-DDA.git
   cd MassNet-DDA

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows

   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

2. **Making Changes**
   - Write tests for new features
   - Ensure all tests pass locally
   - Run code quality checks
   - Update documentation as needed

3. **Pre-commit Checks**
   ```bash
   # Run all checks
   pytest tests/
   flake8 XuanjiNovo tests
   black --check XuanjiNovo tests
   isort --check-only XuanjiNovo tests
   mypy XuanjiNovo
   ```

## Test Coverage

We aim to maintain high test coverage for critical components:
- Model architecture and initialization
- Training and inference pipelines
- Data processing utilities
- CUDA-specific functionality

Current test coverage can be viewed:
- Locally: Generate HTML report with `pytest --cov=XuanjiNovo --cov-report=html`
- Online: Visit our Codecov dashboard

## Adding New Tests

When adding new features or fixing bugs:
1. Create corresponding test file in `tests/`
2. Use appropriate fixtures from `conftest.py`
3. Include both positive and negative test cases
4. Test edge cases and error conditions
5. Add CUDA-specific tests where relevant

## Debugging Tests

For failing tests:
```bash
# Run with detailed output
pytest -vv tests/

# Run specific failing test with debug output
pytest tests/test_model.py::test_name -vv

# Drop into debugger on failure
pytest --pdb tests/
```
