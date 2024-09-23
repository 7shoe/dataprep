import argparse
from mvp_mdpi_utils import MDPI_MVP

def main(i_start):
    mdpi_arx = MDPI_MVP(i_start=i_start)

    # download content
    mdpi_arx.get_arxiv_articles_with_html()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process an integer.')
    parser.add_argument('-i', '--i_start', type=int, default=0, help='Integer index by which BioRXiv database lookup is shifted to')

    args = parser.parse_args()
    main(args.i_start)