from mvp_nature_utils import Nature_MVP
import socket

def main():
    n = Nature_MVP(crawl_delay=4)
    n.complete_database_by_augmenting_pdf_urls()


if __name__=='__main__':
    main()
