import configparser
import argparse


def load_configs():
    parser = argparse.ArgumentParser(description="Override configuration parameters via command line.")

    # Define command-line arguments to override configuration values
    parser.add_argument('--configs', help="The filepath of the .ini configuration file", required=True)
    parser.add_argument('--token', help="The HF token with read and write permissions", required=True)

    args = parser.parse_args()

    configs = configparser.ConfigParser()
    configs.read(args.configs)

    configs.set('TRAIN','token',args.token)

    return configs





