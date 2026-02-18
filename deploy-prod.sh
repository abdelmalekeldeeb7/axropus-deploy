#!/bin/bash
set -euo pipefail

# Deploy Axropus to production

echo "Building frontend..."
cd /home/korith/axropus-cloud/frontend
npm run build

echo "Frontend built. Deploy dist/ to Vercel or Netlify."
echo "Backend: deploy with Docker or Railway/Fly.io/Render."

echo ""
echo "Production checklist:"
echo "  [ ] Frontend deployed to Vercel"
echo "  [ ] Backend deployed to Railway/Fly.io/Render"
echo "  [ ] Domain configured: axropus.com → frontend"
echo "  [ ] Domain configured: api.axropus.com → backend"
echo "  [ ] SSL certificates active"
echo "  [ ] Environment variables set in production"
echo "  [ ] Database migrated"
echo ""
