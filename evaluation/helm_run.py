import subprocess
import argparse

parser = argparse.ArgumentParser(description="Execute evaluation run command after login to Hugging Face.")

parser.add_argument("--token", type=str, default=None, help="The HuggingFace token to use.")


args = parser.parse_args()

huggingface_cli = "huggingface-cli login --token {}".format(args.token)
helm_run = helm_run = "helm-run --run-entries vqa:dataset=test,model=mirodavide/vlm-vqa --suite my-suite --max-eval-instances 10 --disable-cache"


try:
    subprocess.run(huggingface_cli, shell=True, check=True)
    subprocess.run(helm_run, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Command execution failed with error: {e}")