import os
import yaml
import glob
import subprocess

# Define the directory containing your pickle files and the path to the YAML config file
data_dir = "./picklefiles/"
config_path = "./config/lamino_chip.yaml"  # Update with your actual path to the YAML file
output_dir = "./testLog/"

# Get a list of all pickle files in the data directory
pickle_files = glob.glob(os.path.join(data_dir, "*.pickle"))

# Load the initial YAML configuration
with open("./config/lamino_chip.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Iterate over each pickle file
for pickle_file in pickle_files:
    # Update expname and datadir in the configuration
    expname = os.path.splitext(os.path.basename(pickle_file))[0]
    config['exp']['expname'] = expname
    config['exp']['datadir'] = pickle_file
    config['exp']['expdir'] = os.path.join(output_dir, expname)

    # Save the updated config to a temporary YAML file
    temp_config_path = "./config/temp_config.yaml"  # A temporary config file path
    with open(temp_config_path, 'w') as temp_file:
        yaml.dump(config, temp_file)
    
    # Run the training script with the updated config
    try:
        subprocess.run(["python", "train.py", "--config", temp_config_path], check=True)
        print(f"Completed run for: {expname}")
    except subprocess.CalledProcessError as e:
        print(f"Error in running the program for {expname}: {e}")
    
# Clean up the temporary file if needed
if os.path.exists(temp_config_path):
    os.remove(temp_config_path)