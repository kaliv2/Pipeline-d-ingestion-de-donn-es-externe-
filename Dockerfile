FROM openjdk:17-slim

RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install --no-cache-dir pyspark==3.5.1 pandas configparser requests beautifulsoup4

WORKDIR /app

# Copy the entire project structure
COPY . .

ENV ENV=local

# Create a proper entry point script
CMD ["python3", "scripts.processing.data_processing_job"]