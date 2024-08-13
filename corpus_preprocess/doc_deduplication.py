# !/usr/bin/env python3
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
import sys
import time as t

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import HashingTF, Tokenizer, MinHashLSH, NGram
from pyspark.sql.functions import lit
from transformers import AutoTokenizer

from utils.utils import get_spark_session
from utils.utils import save_hdfs

bert_tokenizer = AutoTokenizer.from_pretrained("./bert-base-multilingual-cased/")


@F.udf(returnType=T.BooleanType())
def is_empty_list(data):
    return len(data) != 0


@F.udf(returnType=T.ArrayType(T.StringType()))
def tokenizer_text(text, max_length=4096):
    text = text.lower()
    return bert_tokenizer.tokenize(text)[:max_length]


def get_preprocess_corpus(spark, input_path):
    """
    get preprocess corpus
    """
    # df = spark.read.option('multiline', 'true').format("json").load(input_path)
    df = spark.read.format("json").load(input_path)
    return df


def show_duplicate_records(df, columns):
    not_duplicate_records = df.groupBy(columns).count().where('count = 1').drop('count')
    duplicate_records = df.join(not_duplicate_records, on=columns, how='left_anti')
    print("=================================start=================================")
    duplicate_records.select("hashes", "text").sort(df.hashes.asc(), df.text.asc()).show(n=1000, truncate=False)
    print("=================================end=================================")


def drop_duplicate_hash_text(df, n_gram=2):
    start = t.time()
    # df = df.withColumn("words_list", tokenizer_text(df.text))

    # n gram
    df = df.withColumn("words", tokenizer_text(df.text))
    ngram = NGram(n=n_gram, inputCol="words", outputCol="words_list")
    df = ngram.transform(df)

    # 去除全0向量
    df = df.filter(is_empty_list("words_list"))

    hashing_tf = HashingTF(inputCol="words_list", outputCol="raw_features", numFeatures=120000*1200)
    featured_data = hashing_tf.transform(df)

    mh = MinHashLSH(inputCol="raw_features", outputCol="hashes", numHashTables=128)
    model = mh.fit(featured_data)
    df = model.transform(featured_data)
    # show_duplicate_records(df, "hashes")
    df = df.dropDuplicates(["hashes"])
    df = df.drop(*["words", "words_list", "raw_features", "hashes"])
    end = t.time()
    print("去重数据耗时: %.2f 秒" % (end - start))
    return df


def preprocess_corpus(spark, data_set, input_path, save_path):
    df = get_preprocess_corpus(spark, input_path)
    # df = df.select("text").withColumn("source", lit(data_set))
    raw_count = df.count()
    print(f"Raw: {raw_count}")
    # 删除篇章间重复文章
    df = drop_duplicate_hash_text(df)
    drop_duplicated_text_count = df.count()
    print(f"drop_duplicated_text_count: {drop_duplicated_text_count}")
    # 保存数据
    # save_hdfs(df, save_path, "日语训练集构建结果保存hdfs")


def main():
    dataset_mapping = {
        "merge_ja": [
            "/ipcs/gravity/yeqingzhao/ja_open_oscar_20240306/*",
            # "/ipcs/gravity/yeqingzhao/c4_ja_data_20240229/*",
            # "/ipcs/gravity/yeqingzhao/wiki_ja_data_20240227/*"
        ],
    }
    # 初始化spark session
    print("初始化spark session")
    spark = get_spark_session()
    for dataset, path in dataset_mapping.items():
        print(f"dataset: {dataset}")
        print(f"input_path: {path}")
        save_path = f"/ipcs/gravity/yeqingzhao/ja_c4_oscar_wiki_data_20240307"
        print(f"save_path: {save_path}")
        try:
            preprocess_corpus(spark, dataset, path, save_path)
        except Exception as e:
            print("ERROR", dataset, e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
