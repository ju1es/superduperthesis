def print_usage():
    print('Improper arguments!\n'
          'Run as: python app.py [-m mode] [-c config] [-t transform] [-model model]\n'
          '|  mode = preprocess | train | eval \n'
          '|  config = config-1 | config-2\n'
          '|  transform = logfilt | hcqt | wavenet\n'
          '|  model = baseline | magenta | ASDR\n\n'
          'Ex: python app.py -m preprocess -c config2 -t logfilt -model baseline\n')


def print_bad_config():
    print('ERROR: Dataset config doesn\'t exist.\n')


def print_bad_transform():
    print('ERROR: Transform type doesn\'t exist.\n')


def print_bad_model():
    print('ERROR: Model type doesn\'t exist.\n')


def print_no_data():
    print('ERROR: Data doesn\'t exist. Run preprocessing.\n')


def is_valid_args(config, args):
    result = True
    if args.dataset_config not in config['DATASET_CONFIGS']:
        print_bad_config()
        result = False
    elif args.transform_type not in config['TRANSFORMS']:
        print_bad_transform()
        result = False
    elif args.model not in config['MODELS']:
        print_bad_model()
        result = False
    return result