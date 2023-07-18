"""
Available fields to filter:
for laion-art:
['URL',
 'TEXT',
 'WIDTH',
 'HEIGHT',
 'similarity',
 'LANGUAGE',
 'hash',
 'pwatermark',
 'punsafe',
 'aesthetic']
"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import rand
from pyspark.sql import functions


# filter conditions for sd 2.1 base training
filter_width=512
filter_height=512
lang = 'en'
filter_aes = 4.5
filter_punsafe = 0.98 # used in sd2.1 base training

def filter_metadata(data_path_or_dir, width=None, height=None, num_repartitions=1, lang='en', output_dir=None):
    spark = SparkSession.builder.config("spark.driver.memory", "16G") .master("local[16]").appName('spark-stats').getOrCreate() 
    df = spark.read.parquet(data_path_or_dir)
    num_ori = df.count() 
    print("Num samples: ", num_ori)
    print("Examples: ", df.show(10))
    print("Availabe fields to filter: ", df.schema.names)

    if "WIDTH" in df.schema.names:
        df = df.filter((df.WIDTH >= filter_width) & (df.HEIGHT >= filter_height) & (df.LANGUAGE == lang) & (df.punsafe <= filter_punsafe) & (df.aesthetic>=filter_aes))
    else:
        df = df.filter((df.width >= filter_width) & (df.height >= filter_height) & (df.language == lang) & (df.punsafe <= filter_punsafe) & (df.aesthetic>=filter_aes))
    df = df.orderBy(rand()) # this line is important to have a shuffled dataset
    df.repartition(num_repartitions).write.parquet(output_dir)
    num_after_filtered = df.count()
    print("Num samples after filtering: ", num_after_filtered)
    print(df.select(functions.min("WIDTH")).collect())
    print(df.select(functions.min("HEIGHT")).collect())


def check_filtered_metadata(data_dir):
    spark = SparkSession.builder.config("spark.driver.memory", "16G") .master("local[16]").appName('spark-stats').getOrCreate() 
    df = spark.read.parquet(data_dir)
    num_samples = df.count()
    print("Num samples : ", num_samples)
    print(df.select(functions.min("WIDTH")).collect())
    print(df.select(functions.min("HEIGHT")).collect())
    #print(df.select(functions.min("aesthetic")).collect())

if __name__ == '__main__':
    data_path_or_dir='/data3/datasets/laion_art_metadata'
    num_repartitions = 1
    output_dir = '/data3/datasets/laion_art_metadata_filtered'
    #filter_metadata(data_path_or_dir, filter_width, filter_height, num_repartitions, lang, output_dir=output_dir)
    check_filtered_metadata(output_dir)
   

