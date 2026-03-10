# Enter project directory
cd /opt/clipscut/workers

# Clone your repo (or copy the workers directory)
git pull https://github.com/ElevenMou/workers.git

# Build and run the Docker containers
docker compose up -d --build

# Clean old Docker images
docker system prune -af
docker image prune -af

# Check logs (optional)
docker compose logs -f
