from mvp_bmc_utils import BMC_MVP
import socket
import argparse

def main(modulo):
    nat = BMC_MVP(modulo, crawl_delay=10.0)
    nat.get_arxiv_articles_with_html()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', '--modulo', type=int, default=0, help='An integer for the modulo operation')

    args = parser.parse_args()
    main(args.modulo)