import argparse
from mvp_medrxiv_utils import MedRXiV_MVP

def main(i_start):
    bio_arx = MedRXiV_MVP(i_start=i_start)

    # download content
    bio_arx.get_arxiv_articles_with_html()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process an integer.')
    parser.add_argument('-i', '--i_start', type=int, default=0, help='Integer index by which MedRXiv database lookup is shifted to')

    args = parser.parse_args()
    main(args.i_start)