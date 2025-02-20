# Python 3.6 min

import json
import re
import argparse

# CONSTS
DEFAULT = 'default'
VERSION = 'version'


def _remove_equal(line):
    return re.sub('[=]', '', line)


def pipenvdiff():
    parser = argparse.ArgumentParser(description='Compare two Pipfile.lock files')
    parser.add_argument('file1', type=str, nargs='+', help='File 1 (the old one) to be compared')
    parser.add_argument('file2', type=str, nargs='+', help='File 2 (the new one) to be compared')

    args = parser.parse_args()
    file_1, file_2 = args.file1[0], args.file2[0]

    with open(file_1) as data_file_1, open(file_2) as data_file_2:
        data_1_loaded, data_2_loaded = json.load(data_file_1), json.load(data_file_2)

    packages_1, packages_2 = data_1_loaded[DEFAULT], data_2_loaded[DEFAULT]

    # Naive way to do it, not optimal one
    for package_name, package_info in packages_1.items():
        if package_name in packages_2.keys():
            p1_version = _remove_equal(package_info[VERSION])
            p2_version = _remove_equal(packages_2[package_name][VERSION])
            if not p1_version == p2_version:
                print(f'{package_name}: {p1_version} -> {p2_version}')


if __name__ == "__main__":
    pipenvdiff()
