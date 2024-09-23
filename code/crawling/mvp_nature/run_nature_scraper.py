from mvp_nature_utils import Nature_MVP
import socket
import argparse

def main(modulo):
    nat = Nature_MVP(modulo)
    nat.get_arxiv_articles_with_html()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', '--modulo', type=int, default=0, help='An integer for the modulo operation')

    args = parser.parse_args()
    main(args.modulo)

    