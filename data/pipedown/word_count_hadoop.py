from mrjob.job import MRJob


class WordCountMRJob(MRJob):

    def mapper(self, _, line):
        for word in line.strip().split():
            yield word, 1

    def reducer(self, word, counts):
        yield word, sum(counts)


if __name__ == '__main__':
    WordCountMRJob.run()