import subprocess
import argparse

parser = argparse.ArgumentParser(description="Execute evaluation run command after login to Hugging Face.")

parser.add_argument("model", type=str, default="meta-llama/Meta-Llama-3-8B",help="The HuggingFace model to evaluate.")
parser.add_argument("suite", type=str, default="v1",help="The name of the folder that will save the results.")
parser.add_argument("max_eval_instances", type=int, default=10, help="The number of instances to evaluate.")
parser.add_argument("token", type=str, default=None, help="The HuggingFace token to use.")


args = parser.parse_args()

huggingface_cli = "huggingface-cli login --token {}".format(args.token)
helm_run = "evaluation-run --conf-paths run_entries.conf --enable-huggingface-models {} --suite {} --max-eval-instances {}".format(args.model, args.suite, args)


try:
    subprocess.run(huggingface_cli, shell=True, check=True)
    subprocess.run(helm_run, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Command execution failed with error: {e}")