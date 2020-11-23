FROM tensorflow/tensorflow:2.2.0rc1-py3

RUN pip install -U keras cloudml-hypertune

ADD train.py .

ENTRYPOINT ["python3", "train.py"]
