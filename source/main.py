import argparse
from utils.config import *

from agents import *


def main():
    # Parse the path of the json config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = parser.parse_args()

    # Parse the config json file
    config = process_config(args.config)

    # Create the Agent and pass all the configuration for then to run it
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
