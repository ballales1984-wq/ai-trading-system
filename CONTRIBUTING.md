# Contributing to AI Trading System

We welcome contributions from the community! Whether you're a developer, trader, or just interested in AI trading systems, there are many ways to help improve this project.

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (optional, for development)

### Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/ballales1984-wq/ai-trading-system.git
   cd ai-trading-system
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install Node.js dependencies:

   ```bash
   cd frontend
   npm install
   ```

4. Start the development servers:

   ```bash
   # Backend
   uvicorn app.main:app --reload
   
   # Frontend
   cd frontend
   npm run dev
   ```

## How to Contribute

### 1. Report Issues

If you encounter bugs or have feature requests, please create an issue using our issue templates.

### 2. Fix Bugs

Look for issues with the "bug" label and submit a pull request with your fix.

### 3. Add Features

We welcome new features! Please discuss your idea in an issue before implementing it.

### 4. Improve Documentation

Documentation improvements are always appreciated. This includes README updates, code comments, and tutorials.

### 5. Test Coverage

Help us improve our test coverage by writing new tests for uncovered modules.

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use TypeScript for frontend development
- Include type hints in Python code
- Write clear, descriptive commit messages

### Testing

- All new code should include tests
- Run `pytest` before submitting changes
- Aim for high test coverage

### Documentation

- Update relevant documentation for your changes
- Include examples where appropriate
- Use clear, concise language

## Project Structure

```
ai-trading-system/
├── app/                    # FastAPI application
│   ├── api/routes/        # REST endpoints
│   ├── core/             # Security, cache, DB
│   ├── execution/        # Broker connectors
│   └── database/         # SQLAlchemy models
├── src/                   # Core trading logic
│   ├── agents/           # AI agents
│   ├── core/             # Event bus, state manager
│   ├── decision/         # Decision engine
│   ├── strategy/         # Trading strategies
│   ├── research/         # Alpha Lab, Feature Store
│   └── external/         # API integrations
├── tests/                # Test suite
├── dashboard/            # Dash dashboard
├── frontend/            # React frontend
├── docker/              # Docker configs
└── infra/               # Kubernetes configs
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help others when you can
- Follow the code of conduct

## Resources

- [Documentation](docs/)
- [API Reference](app/docs)
- [Discord Community](https://discord.gg/aitrading)
- [GitHub Issues](https://github.com/ballales1984-wq/ai-trading-system/issues)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or discussions, please use the GitHub Discussions or join our Discord community.
