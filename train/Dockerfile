FROM pytorch/pytorch

# Copy the all source code into the container.
WORKDIR /app
COPY . /app

USER root

# Run the application.
CMD python train.py --batch-size 32 --epochs 1 --no-cuda --no-mps --save-model 