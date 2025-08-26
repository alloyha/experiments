from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName('WordCount')
sc = SparkContext(conf=conf)

input_file = '/path/to/input/file.txt'
output_file = '/path/to/output/directory/result.txt'

lines = sc.textFile(input_file)
word_counts = lines.flatMap(lambda line: line.split()).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
word_counts.saveAsTextFile(output_file)

sc.stop()