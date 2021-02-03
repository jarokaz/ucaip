
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-3

RUN pip install -U 'h5py<3.0.0'

WORKDIR /

# Copies the trainer code to the docker image.
COPY trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]
