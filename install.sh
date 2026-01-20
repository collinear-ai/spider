#!/bin/bash


echo "--- Setting up Python environment ---"
uv python pin 3.12
uv sync
uv pip install -e .[server]

echo "--- Configuring environment variables ---"
# Replace the empty string below with your actual key if preferred, 
# or keep it as an export command.

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

OMP_NUM_THREADS=50