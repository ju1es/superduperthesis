import os
import fnmatch


test_synthnames = set([
    'ENSTDkCl',
    'ENSTDkAm',
])

train_synthnames = set([
    'StbgTGd2',
    'SptkBGCl',
    'SptkBGAm',
    'AkPnStgb',
    'AkPnCGdD',
    'AkPnBsdf',
    'AkPnBcht'
])


def desugar(c):
    prefix = 'MAPS_MUS-'
    last = c[::-1].find('_')
    pid = c[len(prefix):(-last - 1)]
    return prefix, last, pid


def collect_all_piece_ids(synthnames):
    pids = set()
    for synthname in synthnames:
        for base, dirs, files in os.walk(synthname):
            candidates = fnmatch.filter(files, '*MUS*.flac')
            if len(candidates) > 0:
                for c in candidates:
                    _, _, pid = desugar(c)
                    pids.add(pid)

    return pids


def collect_all_filenames(synthnames, include):
    filenames = []
    for synthname in synthnames:
        for base, dirs, files in os.walk(synthname):
            candidates = fnmatch.filter(files, '*MUS*.flac')
            if len(candidates) > 0:
                for c in candidates:
                    _, _, pid = desugar(c)
                    if pid in include:
                        path, ext = os.path.splitext(c)
                        filenames.append(os.path.join(base, path))
    return filenames


def main():
    # Get train/test folder names
    train_pids = collect_all_piece_ids(train_synthnames)
    test_pids = collect_all_piece_ids(test_synthnames)

    print('len(train_pids)', len(train_pids))
    print('len(test_pids)', len(test_pids))

    train_filenames = collect_all_filenames(train_synthnames, train_pids - test_pids)
    test_filenames = collect_all_filenames(test_synthnames, test_pids)

    valid_filenames = []
    for synthname in train_synthnames:
        for filename in train_filenames:
            if filename.startswith(synthname):
                valid_filenames.append(filename)
                break

    print('len(train_filenames)', len(train_filenames))
    print('len(valid_filenames)', len(valid_filenames))
    print('len(test_filenames)', len(test_filenames))

    with open('icassp2019-non-overlapping/train', 'w') as f:
        f.writelines('\n'.join(train_filenames) + '\n')

    with open('icassp2019-non-overlapping/valid', 'w') as f:
        f.writelines('\n'.join(valid_filenames) + '\n')

    with open('icassp2019-non-overlapping/test', 'w') as f:
        f.writelines('\n'.join(test_filenames) + '\n')


if __name__ == '__main__':
    main()
