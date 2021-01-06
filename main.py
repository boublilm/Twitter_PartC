import search_engine_best
import time
import pandas as pd
import numpy as np
import search_engine_3

if __name__ == '__main__':
    se = search_engine_3.SearchEngine()
    se.build_index_from_parquet(r"C:\Users\maorb\OneDrive\Desktop\Search_Engine_Part_C\Search_Engine-master\data\benchmark_data_train.snappy.parquet")
    df = pd.read_csv(r"C:\Users\maorb\OneDrive\Desktop\Search_Engine_Part_C\Search_Engine-master\data\queries_train.tsv", sep='\t')
    times = []
    for i in df['information_need']:
        start_time = time.time()
        x = se.search(i)
        times.append(time.time()-start_time)
    print(f"Average is: {np.mean(times)}")
    print(f"Max is: {max(times)}")

