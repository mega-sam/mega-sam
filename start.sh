#!/bin/bash

# Default dataset_gs path (relative to the project root)
export DATASET_GS_PATH="../datasets_gs"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset=*)
      export DATASET_GS_PATH="${1#*=}"
      shift
      ;;
    --dataset)
      export DATASET_GS_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--dataset=/path/to/dataset_gs]"
      exit 1
      ;;
  esac
done

# Detect environment and set X11 socket path accordingly
if grep -qi microsoft /proc/version; then
  echo "Running in WSL environment"

  # Detect WSL2 specifically
  if [ -f /proc/sys/fs/binfmt_misc/WSLInterop ]; then
    echo "Detected WSL2"
    export X11_SOCKET_PATH="/mnt/wslg/.X11-unix"
  else
    echo "Detected WSL1. WARNING NOT TESTED!!!!!"
    export X11_SOCKET_PATH="/tmp/.X11-unix"
  fi

else
  echo "Running in native Linux environment"
  export X11_SOCKET_PATH="/tmp/.X11-unix"
fi

# Enable X11 forwarding
xhost + local:docker

# Print configuration
echo "Using X11 socket: $X11_SOCKET_PATH"
echo "Using dataset path: $DATASET_GS_PATH"

# Run the container
docker compose run --rm mega_sam