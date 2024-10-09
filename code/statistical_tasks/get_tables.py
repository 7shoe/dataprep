import argparse
from pathlib import Path
from datatable import TextScoreTable

def main(in_file_name, out_file_name, chunk_size, chunk_index, max_char, root_dir):
    # init
    tst = TextScoreTable(db_src_filename = in_file_name,
                         db_dst_filename = out_file_name,
                         chunk_size=chunk_size,
                         chunk_index=chunk_index,
                         max_char=max_char,
                         root_dir=root_dir)

    # compute
    tst.compute_metrics()

    # DEBUG
    #tst.df.to_csv('./tmp/debug_get_tables.csv', sep='|', index=None) # 
    #tst.df_score.to_csv('./tmp/debug_get_tables_score.csv', sep='|', index=None) 
    
    # store
    tst.save_table()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Process a text score table.")
    parser.add_argument('--in_file_name', type=str, required=True, help='Name of the input CSV file.')
    parser.add_argument('--out_file_name', type=str, required=True, help='Name of the output CSV file.')
    parser.add_argument('--chunk_size', type=int, default=-1, required=False, help='Max. number of rows processed.')
    parser.add_argument('--chunk_index', type=int, default=-1, required=False, help='Index of subframe to be processed.')
    parser.add_argument('--max_char', type=int, default=-1, required=False, help='Max. no. of chars kept in parser/groundtruth text')
    parser.add_argument('--root_dir', type=str, default='/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/database', 
                        help='Directory from which data is sourced and stored into')
    args = parser.parse_args()
    
    # validate input/output filenames
    assert isinstance(args.in_file_name, str) and args.in_file_name.endswith('.csv')
    assert isinstance(args.out_file_name, str) and args.out_file_name.endswith('.csv')
    assert Path(args.root_dir).is_dir(), "Path to root_dir must exist"

    # run
    main(in_file_name=args.in_file_name,
         out_file_name=args.out_file_name,
         chunk_size=args.chunk_size,
         chunk_index=args.chunk_index,
         max_char=args.max_char,
         root_dir=args.root_dir)
