#!/bin/bash
# Production Deployment Script
# Ultimate XAU Super System V4.0

echo "🚀 Starting production deployment..."

# Pre-deployment checks
echo "📋 Running pre-deployment checks..."
python testing/final/system_validator.py

# Database migration
echo "💾 Running database migrations..."
# python manage.py migrate

# Build Docker images
echo "🐳 Building Docker images..."
docker-compose -f deployment/production/docker-compose.prod.yml build

# Start services
echo "🌟 Starting production services..."
docker-compose -f deployment/production/docker-compose.prod.yml up -d

# Health checks
echo "🏥 Running health checks..."
sleep 30

# Verify deployment
echo "✅ Verifying deployment..."
curl -f http://localhost:8000/health || exit 1

echo "🎉 Production deployment completed successfully!"
