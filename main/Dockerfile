# base docker image:
FROM python:3.9

# label the docker image:
LABEL Name="petnet"  

# define working directory inside the docker image:
WORKDIR /src

# Copy everything in the current directory into the docker image working directory.
COPY . .


# Assume requirements.txt was in the current directory, install dependencies that require pip install:
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt
ENV RAIVEN_INPUT_DIR=/input
ENV RAIVEN_OUTPUT_DIR=/output
ENV JOB_ID=0
ENV RUN_ID=0
ENV CONTAINER_NAME=DEBUG

# Run the main python code when the container is started:# 
CMD python main.py
