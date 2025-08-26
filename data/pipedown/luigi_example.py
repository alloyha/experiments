import luigi
import luigi.contrib.hadoop
import luigi.contrib.spark
import luigi.contrib.hive


class WordCountHadoop(luigi.contrib.hadoop.JobTask):
    input_file = luigi.Parameter()
    output_dir = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.output_dir)

    def requires(self):
        return None

    def mapper(self, _, line):
        for word in line.strip().split():
            yield word, 1

    def reducer(self, key, values):
        yield key, sum(values)

    def mapper_final(self):
        yield None, None

    def reducer_final(self):
        yield None, None

    def run(self):
        self.run_hadoop_job(
            input_paths=self.input_file,
            output_path=self.output().path,
            mapper=self.mapper,
            reducer=self.reducer,
            mapper_final=self.mapper_final,
            reducer_final=self.reducer_final
        )


class WordCountSpark(luigi.contrib.spark.SparkSubmitTask):
    input_file = luigi.Parameter()
    output_file = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def app(self):
        return 'word_count_spark.py'

    def requires(self):
        return None

    def app_options(self):
        return [self.input_file, self.output().path]


class WordCountHive(luigi.contrib.hive.HiveQueryTask):
    input_file = luigi.Parameter()
    output_table = luigi.Parameter()

    def output(self):
        return luigi.contrib.hive.HiveTarget(self.output_table)

    def requires(self):
        return None

    def query(self):
        return f'''
            CREATE TABLE IF NOT EXISTS {self.output_table} (word STRING, count INT);
            
            LOAD DATA LOCAL INPATH '{self.input_file}' OVERWRITE INTO TABLE {self.output_table};
            
            INSERT OVERWRITE TABLE {self.output_table}
            SELECT word, COUNT(*) AS count
            FROM {self.output_table}
            GROUP BY word;
        '''


if __name__ == '__main__':
    input_file = 'input_file.txt'
    output_dir = '/path/to/output/directory'
    output_table = 'word_counts'

    luigi.build([
        WordCountHadoop(input_file=input_file, output_dir=output_dir),
        WordCountSpark(input_file=input_file, output_file=output_dir + '/result.txt'),
        WordCountHive(input_file=input_file, output_table=output_table)
    ], local_scheduler=True)