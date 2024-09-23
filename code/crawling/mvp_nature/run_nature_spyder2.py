from nature_urls import nature_urls
from mvp_nature_utils import Nature_Spyder2
import pandas as pd
from pathlib import Path

def main():
    # meta
    store_path = Path('./registry/new_nature_db.csv')
    assert store_path.parent.is_dir(), "Parent directory must exist"
    
    # spyder
    ns2 = Nature_Spyder2(nature_urls)
    ns2.scrape_all()
    html_urls = list(set(ns2.html_urls))
    df = pd.DataFrame({'html_url' : html_urls})

    # (append/create) store
    if store_path.is_file():
        df.to_csv(store_path, index=None, header=False, sep='|', mode='a')
    else:
        df.to_csv(store_path, index=None, sep='|')
    pass

if __name__=='__main__':
    main()
    