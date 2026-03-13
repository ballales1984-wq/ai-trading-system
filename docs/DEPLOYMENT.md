# AI Trading System - Deployment Guide

This guide covers how to deploy the AI Trading System to various platforms including Docker, Render, and Vercel.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Deployment](#docker-deployment)
3. [Render Deployment](#render-deployment)
4. [Vercel Deployment](#vercel-deployment)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **CPU**: 2+ cores (4+ recommended for production)
- **RAM**: 4GB+ (8GB+ recommended for production)
- **Storage**: 10GB+ free space
- **Network**: Stable internet connection

### Software Requirements

- **Python**: 3.11+
- **Node.js**: 18+
- **Docker**: 24+ (for Docker deployment)
- **Git**: Latest version

## Docker Deployment

### Quick Start

```bash
# Clone the repository
git clone https://github.com/ballales1984-wq/ai-trading-system.git
cd ai-trading-system

# Start all services
docker-compose up -d

# Wait for services to start (2-3 minutes)
# Then open http://localhost:8000
```

### Services Included

- **API Server**: FastAPI on port 8000
- **Frontend**: React on port 5173
- **Database**: PostgreSQL on port 5432
- **Cache**: Redis on port 6379
- **Dashboard**: Dash on port 8050

### Configuration

Create a `.env` file in the root directory:

```bash
# Exchange API Keys
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret
BYBIT_API_KEY=your_bybit_key
BYBIT_SECRET_KEY=your_bybit_secret

# Market Data APIs
COINMARKETCAP_API_KEY=your_cmc_key
ALPHA_VANTAGE_API_KEY=your_av_key
```

## Render Deployment

### One-Click Deployment

1. Click the "Deploy to Render" button in the README
2. Connect your GitHub repository
3. Configure environment variables
4. Deploy!

### Manual Deployment

```bash
# Install Render CLI
npm install -g @renderhq/cli

# Deploy to Render
render login
render create web --name ai-trading-system --dockerfile ./Dockerfile.render.optimized
```

### Environment Variables

- `PYTHON_VERSION`: 3.11
- `NODE_VERSION`: 18
- `WEB_CONCURRENCY`: 1

## Vercel Deployment

### One-Click Deployment

1. Click the "Deploy to Vercel" button in the README
2. Connect your GitHub repository
3. Configure build settings
4. Deploy!

### Manual Deployment

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy to Vercel
vercel --prod
```

### Configuration

Create `vercel.json`:

```json
{
  "buildCommand": "cd frontend && npm install && npm run build",
  "installCommand": "cd api && pip install --break-system-packages -r requirements.txt",
  "outputDirectory": "frontend/dist",
  "framework": "vite",
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/api/index"
    }
  ]
}
```

## Configuration

### API Keys

Create a `.env` file with your API keys:

```bash
# Exchange Keys
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret
BYBIT_API_KEY=your_bybit_key
BYBIT_SECRET_KEY=your_bybit_secret

# Market Data Keys
COINMARKETCAP_API_KEY=your_cmc_key
ALPHA_VANTAGE_API_KEY=your_av_key
NEWS_API_KEY=your_news_key
```

### Database Configuration

```bash
# PostgreSQL
DATABASE_URL=postgresql://user:password@localhost:5432/ai_trading

# Redis
REDIS_URL=redis://localhost:6379
```

## Troubleshooting

### Common Issues

#### Port Conflicts

```bash
# Check running processes
netstat -tulpn | grep :8000
netstat -tulpn | grep :5173

# Kill conflicting processes
kill -9 <PID>
```

#### Docker Issues

```bash
# Check Docker status
docker ps

# Restart Docker
docker-compose down
docker-compose up -d
```

#### Build Failures

```bash
# Clear cache
docker system prune -a

# Rebuild
docker-compose build --no-cache
```

### Performance Issues

#### Memory Usage

```bash
# Check memory usage
free -h
docker stats
```

#### CPU Usage

```bash
# Check CPU usage
top
htop
```

## Production Considerations

### Security

- Use HTTPS in production
- Set up proper firewall rules
- Use environment variables for secrets
- Regular security updates

### Monitoring

- Set up health checks
- Monitor resource usage
- Set up logging
- Create backup strategies

### Scaling

- Use load balancers
- Set up auto-scaling
- Monitor performance metrics
- Plan for capacity

## Support

For deployment issues, please:

1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information
4. Join our Discord community for real-time help
