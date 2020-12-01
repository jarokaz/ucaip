
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-6
    
RUN pip install -U cloudml-hypertune

ADD train_eval.py .

ENTRYPOINT ["python3", "train_eval.py"]
