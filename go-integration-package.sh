#!/bin/bash

### USE SUDO TO RUN THIS SCRIPT ###

# check if sudo is used
if [ "$EUID" -ne 0 ]; then
    echo "Please run this script with sudo"
    echo "Exits..."
    exit
fi

script_dir=$(dirname "$0")
env_dir="$script_dir/pacgoc_env"

# create temporary directories
sudo -u $SUDO_USER mkdir -p "recordings" "lightning_logs"

# Set the default argument to pcie
default_source="pcie"
source=${1:-$default_source}

# extract pacgoc_env from pacgoc_env.tar.gz
if [ ! -d "pacgoc_env" ]; then
    sudo -u $SUDO_USER mkdir -p "$env_dir"
    echo "extracting pacgoc_env.tar.gz to $env_dir"
    sudo -u $SUDO_USER tar -xzf pacgoc_env.tar.gz -C "$env_dir"
else
    echo "pacgoc_env already exists, skipping extraction"
fi

# activate pacgoc_env and install pacgoc package
sudo -u $SUDO_USER bash -c "source '$env_dir/bin/activate' && \
    if pip list | grep 'pacgoc'; then \
        echo 'pacgoc package is already installed'; \
    else \
        echo 'installing pacgoc package'; \
        pip install -e .; \
    fi"

# compile pango_pcie into shared object
sudo -u $SUDO_USER bash -c "source '$env_dir/bin/activate' && \
    cd pango_pcie && ./run.sh && cd .."

# set up pcie driver
if [ "$source" = "pcie" ]; then
    cd driver && ./run.sh && cd ..
fi

# run app
echo "running app"
case $source in
"pcie")
    env "PATH=$env_dir/bin:$PATH" "$env_dir/bin/python" app/app.py --source $source
    ;;
"speaker")
    sudo -u $SUDO_USER env "XDG_RUNTIME_DIR=/run/user/1000" "PULSE_RUNTIME_PATH=/run/user/1000/pulse/" "$env_dir/bin/python" app/app.py --source $source
    ;;
*)
    sudo -u $SUDO_USER "$env_dir/bin/python" app/app.py --source $source
    ;;
esac
