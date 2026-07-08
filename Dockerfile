FROM python:3.11

# Set the working directory
WORKDIR /code

# Copy requirements and install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all files to the container
COPY . .

# Run Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
