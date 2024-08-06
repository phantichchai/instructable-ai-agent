# Use the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY /api /app/api
COPY /model /app/model
COPY /data/dataset.py /app/data
COPY /tools/utils.py /app/tools
COPY model_weights.pth /app/

# Expose the port on which the app will run
EXPOSE 8000

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
