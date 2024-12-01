FROM --platform=linux/amd64 python:3.8-slim-buster as build

# Copying the application files to container
COPY . /app

# Setting working directory
WORKDIR /app

# Root privileges granted
USER root

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt



# Command to run once container starts - arguments passed
# no-cuda - to disable cuda training,
# no-mps - to disable MacOS gpu training
# epochs and batchsize
EXPOSE 5000
# ENV FLASK_APP=app.py
CMD ["python", "app.py"]