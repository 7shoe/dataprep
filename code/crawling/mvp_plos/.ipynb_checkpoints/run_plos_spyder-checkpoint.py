from plos_urls import plos_urls
from mvp_plos_utils import PLOS_Spyder
import pandas as pd
import time
import random
from pathlib import Path

def main():
    url_list = plos_urls
    p_dst = Path('./registry/plos_database.csv') # destination path (assumes CSV exists in there)

    # random init strategy to eschew rejected HTTP requests
    for i in range(60):
        random.shuffle(url_list)
        
        # launch spyder
        spyder = PLOS_Spyder(url_list[:2])
        article_urls = spyder.scrape_all()

        # -> DF
        df_plos = pd.DataFrame({'html_url' : spyder.articles_urls})

        # store
        if p_dst.is_file():
            df_plos.to_csv(p_dst, sep='|', mode='a', header=False, index=None)
        else:
            df_plos.to_csv(p_dst, sep='|', index=None)
    
        time.sleep(10)


if __name__=='__main__':
    main()