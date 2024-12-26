#!/bin/bash

# Step 1: Start the database containers
# Check for the first argument
if [ "$1" = "local-embedding" ]; then
  echo "Setting up local embedding server..."
   docker-compose --profile local-embedding up -d
else
  echo "Skipping embedding server setup."
  docker-compose up -d
fi

# Step 2: Wait for the databases to initialize
echo "Waiting for MongoDB and PostgreSQL to be ready..."
sleep 10  # Adjust this if databases need more time to initialize

# Step 3: Create and activate Conda environment
ENV_NAME="test"

git clone https://github.com/hung20gg/llm.git

echo "Creating Conda environment..."
conda create -y -n $ENV_NAME python=3.10
conda activate $ENV_NAME

# Step 4: Install dependencies using pip
echo "Installing dependencies..."
pip install -r requirements.txt

# Step 5: Run setup.py
if [ -f "setup.py" ]; then
  echo "Running setup.py..."
  python3 setup.py "$@"
else
  echo "setup.py not found, skipping..."
fi

# Step 6: Run test.py
if [ -f "test.py" ]; then
  echo "Running test.py..."
  python3 test.py
else
  echo "test.py not found, skipping..."
fi

# Step 7: Run Streamlit application
if [ -f "home.py" ]; then
  echo "Starting Streamlit app..."
  streamlit run home.py
else
  echo "home.py not found, skipping Streamlit launch..."
fi
