FROM python:3.10.6-slim-bullseye
#USER root
RUN apt-get update && apt-get install -y \
    git \
    ssh \
    octave \ 
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

RUN mkdir -p /root/.ssh

WORKDIR /simulation

COPY system.json .
COPY components.json .
COPY LocalFeeder LocalFeeder
<<<<<<< HEAD
COPY lindistflow_federate lindistflow_federate
COPY README.md .
COPY measuring_federate measuring_federate
COPY wls_federate wls_federate
=======
>>>>>>> 15a4e55 (upload)
COPY recorder recorder
COPY dsse_federate dsse_federate
COPY measuring_federate measuring_federate
COPY dopf.ipynb .

RUN mkdir -p outputs build

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN echo "addpath(genpath('/simulation/dsse_federate'))" >> /root/.octaverc

EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--no-browser"] 
