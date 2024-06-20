import os
from scripts.preprocess import run as preprocess_run
from scripts.train import run as train_run

input_dir = 'data/input'
output_dir = 'data/output'

# Ensure the directories exist
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Step 1: Preprocess the data
preprocess_run(input_dir, output_dir)

# Step 2: Train the model
input_shape = (256, 256, 3)
num_classes = 5
train_run(output_dir, input_shape, num_classes)
