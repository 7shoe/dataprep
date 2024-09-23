import argparse
import time
from mvp_arxiv_utils import ArXiV_MVP

def main(modulo):
    arx = ArXiV_MVP(modulo)

    for search_word in arx.search_words:
        time.sleep(20)
        outlist = arx.get_arxiv_articles_with_html(query=search_word, max_results=100, download=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', '--modulo', type=int, default=0, help='An integer for the modulo operation')

    args = parser.parse_args()
    main(args.modulo)
