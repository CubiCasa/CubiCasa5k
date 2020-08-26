FROM anibali/pytorch:cuda-9.0

RUN sudo apt-get update
RUN sudo apt-get upgrade -y
RUN sudo apt-get install -y \
        build-essential 

COPY requirements.txt /app/.

RUN pip install -r requirements.txt --ignore-installed

