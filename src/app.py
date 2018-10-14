import preprocess as pre
import train
import evaluate as eval
import sys
import argparse
from configs import CONFIG
from lib import errors as e

if __name__ == "__main__":
    print("\nTranscribe all the music...\n")

    num_of_args = len(sys.argv)
    if num_of_args != 9:
        e.print_usage()
        sys.exit()

    arg_parser = argparse.ArgumentParser(description='Get run specs.')
    arg_parser.add_argument('-m', dest='mode', required=True)
    arg_parser.add_argument('-model', dest='model', required=True)
    arg_parser.add_argument('-c', dest='dataset_config', required=True)
    arg_parser.add_argument('-t', dest='transform_type', required=True)
    args = arg_parser.parse_args()

    experiment_id = args.dataset_config + "_" + args.transform_type + "_" + args.model
    if args.mode == 'preprocess' and e.is_valid_args(CONFIG, args):
        pre.run(CONFIG, args, experiment_id)
    elif args.mode == 'train' and e.is_valid_args(CONFIG, args):
        train.run(CONFIG, args, experiment_id)
    elif args.mode == 'evaluate' and e.is_valid_args(CONFIG, args):
        eval.run(CONFIG, args, experiment_id)
    else:
        e.print_usage()
    sys.exit()

