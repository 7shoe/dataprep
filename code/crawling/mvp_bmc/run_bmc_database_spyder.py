import pandas as pd
from pathlib import Path
import re
from bmc_urls import bmc_urls

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import time
import socket

from bmc_urls import bmc_urls

from mvp_bmc_utils import BMC_Spyder, BMC_MVP

def main():
    ## Find more downloadable URLs
    bmc = BMC_Spyder(url_list=[]) # just post-processing now
    bmc.complete_database_by_augmenting_pdf_urls()

if __name__=='__main__':
    main()