from mvp_biorxiv_utils import BioRXiV_Meta_Creator
import argparse

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Process an integer.')
    parser.add_argument('-i', '--i', type=int, default=0, help='Integer index by which BioRXiv database lookup is shifted to')
    args = parser.parse_args()
    
    meta = BioRXiV_Meta_Creator()
    meta.store_biorxiv_meta(i=args.i)

if __name__=='__main__':
    main()