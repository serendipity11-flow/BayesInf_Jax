import subprocess
import sys

# Get the path to the training script
module_name = "bayesinf.models.train_parameter_identification_model"

flow_type = 'RealNVP'
# 1. Run with --force for multiple dt0 values
dt0 = 0.05
noise_values = [0.01, 0.02, 0.05, 0.1]
for noise_std in noise_values:
    cmd = [
        sys.executable,
        "-m", module_name,
        "--test",
        f"--flow_type={flow_type}",
        "--force",
        f"--dt0={dt0}",
        f"--noise_std={noise_std}"
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=".")
    if result.returncode != 0:
        print(f"Error running dt0={dt0}")
        sys.exit(1)

# # 2. Run with --linear
# cmd = [
#     sys.executable,
#     "-m", module_name,
#     "--test",
#     f"--flow_type={flow_type}",
#     "--linear",
#     "--dt0=0.05"
# ]
# print(f"\nRunning: {' '.join(cmd)}")
# result = subprocess.run(cmd, cwd=".")
# if result.returncode != 0:
#     print("Error running linear test")
#     sys.exit(1)

# # 3. Run with --multi_modal
# cmd = [
#     sys.executable,
#     "-m", module_name,
#     "--test",
#     f"--flow_type={flow_type}",
#     "--multi_modal",
#     "--dt0=0.05"
# ]
# print(f"\nRunning: {' '.join(cmd)}")
# result = subprocess.run(cmd, cwd=".")

if result.returncode != 0:
    print("Error running multi_modal test")
    sys.exit(1)

print("\nAll runs completed successfully!")