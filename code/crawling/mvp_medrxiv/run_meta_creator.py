from mvp_medrxiv_utils import MedRXiV_Meta_Creator
import argparse

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Process an integer.')
    parser.add_argument('-i', '--i', type=int, default=0, help='Integer index by which BioRXiv database lookup is shifted to')
    args = parser.parse_args()
    
    meta = MedRXiV_Meta_Creator()
    meta.store_medrxiv_meta(i=args.i)

if __name__=='__main__':
    main()
