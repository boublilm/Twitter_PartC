import search_engine_best
import time
import pandas as pd
import numpy as np
import search_engine_3

if __name__ == '__main__':
    se = search_engine_3.SearchEngine()
    se.build_index_from_parquet(r"C:\Users\maorb\OneDrive\Desktop\Search_Engine_Part_C\Search_Engine-master\data\benchmark_data_train.snappy.parquet")
    querys = ['Dr. Anthony Fauci wrote in a 2005 paper published in Virology Journal that hydroxychloroquine was effective in treating SARS.','The seasonal flu kills more people every year in the U.S. than COVID-19 has to date.','The coronavirus pandemic is a cover for a plan to implant trackable microchips and that the Microsoft co-founder Bill Gates is behind it','Herd immunity has been reached.','Children are “almost immune from this disease.”']
    for i in querys:
        print(i)
        #x = se.search(i)
        #print(x[1][:5])


