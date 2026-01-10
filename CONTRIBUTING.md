# Contributing to Examiner

Thank you for your interest in contributing to Examiner! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and constructive. We're building this together.

## How to Contribute

### Reporting Issues

- **Bug Reports**: Describe the issue, steps to reproduce, expected behavior, actual behavior, and environment
- **Feature Requests**: Explain the use case, benefits, and any alternative approaches considered
- **Questions**: Use GitHub Discussions for general questions

### Submitting Changes

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make changes** with clear, descriptive commits
4. **Test** your changes (verify with `verify_system.py`)
5. **Document** your changes (docstrings, comments, README updates)
6. **Push** to your fork: `git push origin feature/your-feature-name`
7. **Create a Pull Request** with detailed description

### Coding Standards

- Follow PEP 8 style guide
- Add docstrings to all functions and classes
- Include type hints where practical
- Keep functions focused and testable
- Use descriptive variable names

### Testing

Run before submitting PR:
```bash
python verify_system.py      # System verification
python check_gpu.py          # GPU check
python architecture_auditor.py  # Architecture validation
```

### Documentation

- Update relevant markdown files
- Add docstrings to new functions
- Include examples for new features
- Update `README.md` if adding major features

## Development Areas

### Priority 1: High Impact
- Multi-corpus support (multiple knowledge bases)
- Automated evaluation metrics (grounding quality)
- Training stability improvements
- Memory optimization

### Priority 2: Feature Enhancement
- Multi-model support (Llama, Mistral, etc.)
- Advanced inference options (beam search, top-k sampling)
- Corpus update mechanisms (incremental learning)
- Visualization tools

### Priority 3: Infrastructure
- Docker containerization
- API server (FastAPI)
- Web UI
- Deployment guides

## Questions or Need Help?

- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and ideas
- **Pull Requests**: Feel free to comment on PRs for feedback

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

Thank you for contributing to making Examiner better! 🙏
