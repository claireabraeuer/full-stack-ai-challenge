# Docker Deployment Guide

This guide explains how to run the Intelligent Support System using Docker.

## Quick Start

### 1. Build and Start All Services

```bash
docker-compose up -d
```

This starts:
- **API Server** (port 8000) - Main application
- **MLflow Server** (port 5000) - Experiment tracking UI
- **Prometheus** (port 9090) - Metrics monitoring

### 2. Verify Services

```bash
# Check container status
docker-compose ps

# View API logs
docker-compose logs -f api

# Health check
curl http://localhost:8000/health
```

### 3. Access Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Metrics**: http://localhost:8000/metrics

---



## Common Commands

### Start Services

```bash
# Start all services
docker-compose up -d

# Start only API
docker-compose up -d api

# Build and start (after code changes)
docker-compose up -d --build
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 api
```

### Stop Services

```bash
# Stop all
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop specific service
docker-compose stop api
```

### Debugging

```bash
# Enter API container
docker-compose exec api bash

# Check container resource usage
docker stats

# Inspect container
docker-compose exec api python -c "import sys; print(sys.path)"
```
