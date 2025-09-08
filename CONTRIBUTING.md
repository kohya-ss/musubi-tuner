# Contributing to Musubi Tuner

Thank you for your interest in contributing to Musubi Tuner! We welcome contributions from the community and are excited to work with you to make this project even better.

## Table of Contents

- [Getting Started](#getting-started)
- [Before You Contribute](#before-you-contribute)
- [How to Contribute](#how-to-contribute)
  - [Reporting Issues](#reporting-issues)
  - [Suggesting Features](#suggesting-features)
  - [Contributing Code](#contributing-code)
- [Development Setup](#development-setup)
- [Code Style and Guidelines](#code-style-and-guidelines)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Licensing and Attribution](#licensing-and-attribution)
- [Community and Support](#community-and-support)

## Getting Started

Before contributing, please:

1. Read through this contributing guide
2. Review the [README.md](README.md) to understand the project
3. Check the [existing issues](https://github.com/kohya-ss/musubi-tuner/issues) and [discussions](https://github.com/kohya-ss/musubi-tuner/discussions)
4. Set up your development environment

## Before You Contribute

### Important Notes

- This project is under active development with limited maintainer resources
- PR reviews and merges may take time
- Breaking changes may occur as the project evolves
- For questions and general discussion, use [GitHub Discussions](https://github.com/kohya-ss/musubi-tuner/discussions)
- For bug reports and feature requests, use [GitHub Issues](https://github.com/kohya-ss/musubi-tuner/issues)

### Types of Contributions We Welcome

- Bug fixes
- Performance improvements
- Documentation improvements
- New features (with prior discussion)
- Code quality improvements

## How to Contribute

### Reporting Issues

Before creating a new issue:

1. **Search existing issues** to avoid duplicates
2. **Check discussions** as your question might already be answered

When creating a bug report, include:

- **Clear, descriptive title**
- **Detailed description** of the problem
- **Steps to reproduce** the issue
- **Environment details**:
  - Operating System
  - GPU model and VRAM
  - Python version
  - PyTorch version
  - CUDA version
- **Error messages or logs**
- **Expected vs actual behavior**
- **Screenshots or videos** (if applicable)

### Suggesting Features

For feature requests:

1. **Open an issue first** to discuss the feature
2. **Explain the problem** your feature would solve
3. **Describe the proposed solution**
4. **Consider alternatives** and their trade-offs
5. **Wait for feedback** before starting implementation (there's always a chance the PR won't be merged)

For significant features, consider posting in [GitHub Discussions](https://github.com/kohya-ss/musubi-tuner/discussions) first to gather community input.

### Contributing Code

1. **Open an issue** to discuss your proposed changes (unless it's a trivial fix)
2. **Wait for approval** before starting work on significant changes
3. **Fork the repository** and create a feature branch
4. **Make your changes** following our code style guidelines
5. **Test your changes** thoroughly
6. **Submit a pull request**

## Development Setup

### Prerequisites

- Python 3.10 or later
- Git
- CUDA-compatible GPU (for testing GPU features)
- 12GB+ VRAM recommended

### Installation

1. **Fork and clone the repository**:
   ```shell
   git clone https://github.com/your-username/musubi-tuner.git
   cd musubi-tuner
   ```

2. **Set up the development environment**:

   **Option A: Using pip**
   ```shell
   # Create virtual environment
   python -m venv .venv

   # Activate virtual environment
   # On Windows:
   .venv/Scripts/activate
   # On Linux/Mac:
   source .venv/bin/activate

   # Install PyTorch (adjust for your CUDA version)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

   # Install the package in development mode
   pip install -e .

   # Install development dependencies
   pip install --group dev
   ```

	**Option B: Using uv**
   ```shell
   # Install uv if not present
   curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
   # or
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

   # Install dependencies
   uv sync --extra cu128  # or cu124 based on your CUDA version
   ```

3. **Configure Accelerate**:
   ```shell
   accelerate config
   ```

## Code Style and Guidelines

### Python Code Style

This project uses **Ruff** for code linting and code formatting:

- **Line length**: 132 characters
- **Indentation**: 4 spaces
- **Quote style**: Double quotes
- **Target Python version**: 3.10

### IDE

https://docs.astral.sh/ruff/editors/setup/

### Running Code Quality Tools

```shell
# Check code style and potential issues
ruff check

# Auto-fix issues where possible
ruff check --fix

# Format code (note: use ruff for formatting, not black)
ruff format src
```

### Code Guidelines

- **Follow existing patterns** in the codebase
- **Write clear, descriptive variable names**
- **Add type hints** where appropriate
- **Keep functions focused** and reasonably sized
- **Add docstrings** for public functions and classes
- **Handle errors appropriately** - Let unrecoverable errors fail fast; only catch and handle errors you can meaningfully recover from

### Import Organization

- Standard library imports first
- Third-party imports second
- Local imports last
- Use absolute imports when possible

### Code Modification Guidelines

When working with existing code:

- **Maintain compatibility** with existing interfaces
- **Follow the existing module structure**
- **Update relevant documentation** in the `docs/` directory
- **Test across different architectures** if your changes affect multiple architectures and you have the capability to do so

When working with architecture-specific code (HunyuanVideo, Wan2.1/2.2, FramePack, FLUX.1 Kontext, Qwen-Image):

- **Follow naming conventions**: When adding a new architecture, follow the `{arch}_train_network.py` and `{arch}_generate_{type}.py` naming pattern
- **Consider cross-architecture impact** when making changes within shared modules
- **Test with representative models** if possible

## Testing

### Running Tests

```shell
# Run code quality checks
ruff check

# Format code
ruff format src

# Test your changes manually with the relevant scripts
```

### Manual Testing Guidelines

Since this project deals with machine learning models:

1. **Test with small datasets** first
2. **Verify memory usage** is within expected boundaries
3. **Test on different GPU configurations** if possible
4. **Validate output quality** for generation/training features

## Pull Request Process

### Before Submitting

1. **Ensure your branch is up to date** with the main branch
2. **Run code quality tools**:
   ```shell
   ruff check --fix
   ruff format src
   ```
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Write clear commit messages**

### Pull Request Template

When creating a PR, include:

- **Clear title** describing the change
- **Description** of what changed and why
- **Issue reference** (e.g., "Closes #123")
- **Testing performed**
- **Breaking changes** (if any)
- **Documentation updates** (if any)

### Review Process

- Maintainers will review PRs when time permit
- Be patient as reviews may take time due to limited resources
- Address feedback constructively
- Keep discussions focused and professional

## Licensing and Attribution

### Attribution Requirements

When contributing code derived from or inspired by other projects:

1. **Add appropriate license headers** to new files
2. **Include attribution comments** for copied/modified code
3. **Update the LICENSE section on README.md** if introducing new license requirements for new architectures
4. **Document the source** in your pull request description

### Third-Party Code

If your contribution includes third-party code:

1. **Ensure license compatibility** with the project
2. **Include the original license file** or header
3. **Document the source and license** clearly. Incorporate this in your pull request description as well
4. **Fulfill all obligations** from the source license

## Community and Support

### Communication Channels

- **GitHub Discussions**: General questions, ideas, and community interaction
- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions and reviews

### Getting Help

If you need help with:

- **Using the software**: Check [GitHub Discussions](https://github.com/kohya-ss/musubi-tuner/discussions)
- **Development setup**: Create an issue with the "question" label or ask in discussions
- **Contributing process**: Reference this guide or ask in discussions

### Recognition

Contributors are recognized through:

- **Git commit history**
- **Release notes** for significant contributions
- **README acknowledgments** for major features

---

## Final Notes

We appreciate your interest in contributing to Musubi Tuner! This project benefits greatly from community contributions, and we're grateful for your time and effort.

Remember:
- **Start small** with your first contribution
- **Ask questions** if anything is unclear
- **Be patient** with the review process
- **Have fun** building amazing tools!

Thank you for helping make Musubi Tuner better for everyone!
