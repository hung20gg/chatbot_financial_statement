# Base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy application code into the container
COPY . /usr/src/app

# Copy requirements file
COPY requirements.txt /usr/src/app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default Streamlit port
EXPOSE 8501

# Default command to run setup.py, test.py, and Streamlit
CMD ["sh", "-c", "python setup.py && python test.py && streamlit run home.py"]
