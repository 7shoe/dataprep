from meta_utils import Terminator_of_already_parsed_Files
import argparse

def main(subdir_name):
    terminator = Terminator_of_already_parsed_Files(subdir_name)
    terminator.delete_already_parsed_files()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Delete already parsed files for a given subdir_name.')
    parser.add_argument('-s', '--subdir_name', type=str, help="Either 'arxiv', 'biorxiv', 'medrxiv', 'mdpi', 'plos', 'bmc', or 'nature')")

    args = parser.parse_args()
    subdir_name = args.subdir_name
    main(subdir_name)