# Use an official Python runtime as a parent image
FROM python:3.10.12

# Set the working directory in the container
WORKDIR /Users/nikhil/instadeep-test

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Define the command to run when the container starts
CMD ["python", "src/train.py"]