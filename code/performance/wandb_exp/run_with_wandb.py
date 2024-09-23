import subprocess
import wandb
import os

# Initialize wandb
wandb.init(project="your_project_name", name="your_run_name", config={"_stats_sample_rate_seconds": 2})

# Run the shell script
process = subprocess.Popen(['./simple_program.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Capture stdout and stderr
stdout, stderr = process.communicate()

# Log stdout and stderr to wandb
wandb.log({"stdout": stdout.decode('utf-8')})
wandb.log({"stderr": stderr.decode('utf-8')})

# Log the exit status
exit_code = process.returncode
wandb.log({"exit_code": exit_code})

# Mark wandb run as finished
wandb.finish()
