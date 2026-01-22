#!/bin/bash

echo "--- Setting up Python environment ---"
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python pin 3.12
uv sync && uv pip install -e .[server]

echo "--- Configuring environment variables ---"

echo "--- Logging into Weights & Biases ---"
uv run wandb login

echo "--- Setting up Docker ---"
mkdir -p docker
sudo snap install docker
# Note: docker login usually requires interactive input
docker login
sudo groupadd -f docker

# Add the 'ubuntu' user to the docker group
sudo usermod -aG docker ubuntu

# Apply the group changes to the current session
newgrp docker
sudo chmod 666 /var/run/docker.sock

echo "--- Starting training script ---"
uv run scripts/train_on_policy_swe_accelerate.py

export OMP_NUM_THREADS=50
export HF_HOME=/mnt/local/hf_home

##### Docker Storage Fix #####
# Stop Docker to move data directory
sudo systemctl stop docker.socket docker.service

# Create new Docker data directory on /mnt/local (12TB available)
sudo mkdir -p /mnt/local/docker-root && sudo chmod 755 /mnt/local/docker-root

# Create daemon.json (for reference, though snap Docker uses symlink)
sudo mkdir -p /etc/docker
sudo bash -c 'echo "{
  \"data-root\": \"/mnt/local/docker-root\"
}" > /etc/docker/daemon.json'

# For snap-installed Docker, create symlink from default location to /mnt/local
# Backup existing data if it exists
if [ -d /var/snap/docker/common/var-lib-docker ] && [ ! -L /var/snap/docker/common/var-lib-docker ]; then
    sudo mv /var/snap/docker/common/var-lib-docker /var/snap/docker/common/var-lib-docker.backup
fi
# Create symlink to new location
sudo ln -sf /mnt/local/docker-root /var/snap/docker/common/var-lib-docker

# Restart Docker and verify
sudo systemctl start docker.socket docker.service
sleep 3
docker info | grep "Docker Root Dir"

##### Docker Buildx Cache Setup #####
# Create buildx cache directory on /mnt/local
sudo mkdir -p /mnt/local/docker-cache && sudo chmod 777 /mnt/local/docker-cache

# Create buildx builder with cache support (docker-container driver supports cache export)
docker buildx create --name cache-builder --driver docker-container --use 2>/dev/null || \
    docker buildx use cache-builder 2>/dev/null || true

# Bootstrap the builder
docker buildx inspect --bootstrap > /dev/null 2>&1

echo "Docker storage configured: $(docker info | grep 'Docker Root Dir')"
echo "Buildx cache directory: /mnt/local/docker-cache"


##### Docker Start & Login after reboot #####
sudo systemctl restart docker.socket && sleep 2 && sudo systemctl restart docker
# docker login with adit's docker account 🙏