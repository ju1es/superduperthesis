import preprocess as pre
import sys
import argparse
from configs import CONFIG

def print_usage():
    print('Improper arguments!\n'
          'Run as: python app.py [-m mode] [-c config] [-t transform] [-model model]\n'
          '|  mode = preprocess | train (and evaluate)\n'
          '|  config = config-1 | config-2\n'
          '|  transform = logfilt | hcqt | wavenet\n'
          '|  model = baseline | magenta | ASDR\n\n'
          'Ex: python app.py -m preprocess -c config2 -t logfilt -model baseline\n')

def print_bad_config():
    print('Error: Dataset config doesn\'t exist.\n')

def print_bad_transform():
    print('Error: Transform type doesn\'t exist.\n')

def print_bad_model():
    print('Error: Model type doesn\'t exist.\n')

if __name__ == "__main__":
    print("\nTranscribe all the music...\n")

    num_of_args = len(sys.argv)
    if num_of_args != 9:
        print_usage()
        sys.exit()

    arg_parser = argparse.ArgumentParser(description='Get run specs.')
    arg_parser.add_argument('-m', dest='mode', required=True)
    arg_parser.add_argument('-model', dest='model', required=True)
    arg_parser.add_argument('-c', dest='dataset_config', required=True)
    arg_parser.add_argument('-t', dest='transform_type', required=True)
    args = arg_parser.parse_args()

    experiment_id = args.dataset_config + "_" + args.transform_type + "_" + args.model
    if args.mode == 'preprocess':
        if args.dataset_config not in CONFIG['DATASET_CONFIGS']:
            print_bad_config()
        elif args.transform_type not in CONFIG['TRANSFORMS']:
            print_bad_transform()
        elif args.model not in CONFIG['MODELS']:
            print_bad_model()
        else:
            pre.run(CONFIG, args, experiment_id)
    elif args.mode == 'train':
        # To implement
        pass
    else:
        print_usage()
    sys.exit()

