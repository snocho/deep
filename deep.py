import requests
import bs4
import logging
import csv
import re
import pandas as pd
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

class Crawler:
    def __init__(self, input_file, output_file_train, output_file_test):
        self.logger = logging.getLogger('main.' + self.__class__.__name__)
        self.input_file = input_file
        self.output_file_train = output_file_train
        self.output_file_test = output_file_test
        self.start_processing()

    def start_processing(self):
        self.logger.info('Start execution!')
        with open(self.input_file, 'r') as f:
            for line in f:
                result = self.process_url(line)
                self.write_to_file()

    def process_url(self, url): 
        self.reviews = []
        # hopefuly there won't be more than 99 pages of reviews
        for i in range(1, 99):
            # we have to remove last character (new line symbol)
            url = url.replace('\n', '')
            # first page has different url
            if i == 1:
                r = requests.get(url + '/discussion')
            else:
                r = requests.get(url + '/discussion?page={}'.format(i))
            self.logger.debug('Html request for page {}: {}'.format(i, r))
            if r.status_code != 200:
                self.logger.error('server returns wrong code')
                break
            soup = bs4.BeautifulSoup(r.text, 'html.parser')
            # ohiden topicWrapper is the review div's name
            rows = soup.find_all('div', attrs={'class': 'ohidden topicWrapper'})
            self.logger.info('number of reviews found at page {}: {}'.format(i, len(rows)))
            if len(rows) == 0:
                break
            for row in rows:
                review = []
                # sometimes what we are looking for is in the first three fields, but sometimes we have to check more - but to do not wait forever, we pick 10 as arbitrary number of checks
                for counter in range(10):
                    try:
                        item = row.contents[counter].text
                        counter += 1
                        # remove all formatting tags
                        item = ' '.join(item.split())
                        review.append(item)
                    except AttributeError:
                        # exception raise when item has no text representation (ie navigation element or sinlge \n, then we just increment counter to procedee to the next element
                        counter += 1
                    # if review item has three elements: title, score, review text we are done
                    if len(review) == 3:
                        # extract only score from the second element
                        try:
                            m = re.search('ten film na: (\d+)', review[1])
                            review[1] = m.group(1)
                        except AttributeError:
                            # no score in review
                            self.logger.debug('review with no score!')
                            # clear review
                            review = []
                        break
                # append only when review list is complete (has score) and
                # has no info about being blocked
                if review:
                    # if review contains message about being a spoiler - then remove it
                    if len(review[2]) == 65:
                        review[2] == ''
                    self.reviews.append(review)

        self.logger.info('total number of complete reviews to save: {}'.format(len(self.reviews)))

    def write_to_file(self, train_percent=80, test_percent=20):
        if train_percent + test_percent != 100:
            self.logger.critical('Testing and training percents are not equal to 100%: train - {} test - {}'.format(train_percent, test_percent))
            exit(1)
        if train_percent < 80:
            self.logger.warn('Train percent less than 80%: {}, consider increasing this value'.format(train_percent))
        train_len = int((train_percent/100) * len(self.reviews))

        self.logger.info('append results to train file: {}'.format(self.output_file_train))
        review_test = []
        review_train = []

        for num, review in enumerate(self.reviews):
            row = ' '.join([review[0], review[2]])
            row = row.replace('"', '')
            row = row.replace('\'', '')
            row = row.replace('\t', ' ')
            if num < train_len:
                review_train.append([row, review[1]])
            else:
                review_test.append([row, review[1]])

        outfile_test = pd.DataFrame(data=review_test, columns=['review', 'score'])
        outfile_train = pd.DataFrame(data=review_train, columns=['review', 'score'])
        
        outfile_test.to_csv(self.output_file_test, quoting=3, index_label='id', sep='\t')
        outfile_train.to_csv(self.output_file_train, quoting=3, index_label='id', sep='\t')

class BOW:
    def __init__(self, stop_words_file, file_name_train, file_name_test, output_file):
        self.logger = logging.getLogger('main.' + self.__class__.__name__)

        self.output_file = output_file

        self.logger.info('Reading input train data from file: {}'.format(file_name_train))
        self.train = pd.read_csv(file_name_train, header=0, delimiter='\t', quoting=3)

        self.logger.info('Reading input test data from file: {}'.format(file_name_test))
        self.test = pd.read_csv(file_name_test, header=0, delimiter='\t', quoting=3)

        self.logger.info('Reading stop words from file: {}'.format(stop_words_file))
        with open(stop_words_file, 'r') as f:
            self.stop_words = [x.strip() for x in f.readlines()]

        # in order to inform about progress we need the size of the input
        train_len = self.train['review'].size 
        test_len = self.test['review'].size
        self.clean_test = []

        train_len_10percent = int(train_len/10)
        test_len_10percent = int(test_len/10)        
        self.clean_train = []
        
        self.logger.info('Start cleaning train data, number of reviews: {}'.format(train_len))
        for i in range(0, train_len):
            if not i % train_len_10percent:
                self.logger.debug('{:3.0f}% done!'.format(i/train_len * 100))
            self.clean_train.append(self.clean_review(self.train['review'][i]))
        self.logger.debug(' 100% done!')

        self.logger.info('Start cleaning test data, number of reviews: {}'.format(test_len))
        for i in range(0, test_len):
            if not i % test_len_10percent:
                self.logger.debug('{:3.0f}% done!'.format(i/test_len * 100))
            self.clean_test.append(self.clean_review(self.test['review'][i]))
        self.logger.debug(' 100% done!')
        
        # create model
        self.create_bag_of_word()

        # run classification
        self.classify()

        # run tests
        self.run_test()
        
        # write_results_to_file
        self.write_results()

    def clean_review(self, review):
        # remove urls
        text_no_urls = re.sub(r'((http:\/\/|https:\/\/|www):?\S+)', '', review)
        
        # remove Polish accents
        text_no_accents = unidecode(text_no_urls)

        # remove non-letters
        letters_only = re.sub('[^a-zA-Z]', ' ', text_no_accents)

        # convert to lover case
        words = letters_only.lower().split()

        # remove stop words
        stops = set(self.stop_words)
        meaningful_words = [w for w in words if w not in stops]

        # back to strings
        return (' '.join(meaningful_words))

    def create_bag_of_word(self):
        self.logger.info('Create count vectorizer')
        self.vectorizer = CountVectorizer(analyzer='word',
                                          tokenizer=None,
                                          preprocessor=None,
                                          stop_words=None,
                                          max_features=300)

        self.logger.info('Fit transform')
        self.train_data_features = self.vectorizer.fit_transform(self.clean_train)

        self.logger.info('To array..')
        self.train_data_features = self.train_data_features.toarray()

    def classify(self):
        self.logger.info('Create classifier')
        self.forest = RandomForestClassifier(n_estimators=100)

        self.logger.info('Fit classifier')
        self.forest = self.forest.fit(self.train_data_features, self.train['score'])
        
    def run_test(self):
        self.logger.info('Run tests...')
        self.test_data_features = self.vectorizer.transform(self.clean_test)

        self.logger.info('To array...')
        self.test_data_features = self.test_data_features.toarray()

        self.logger.info('Predict...')
        self.result = self.forest.predict(self.test_data_features)

    def write_results(self):
        self.logger.info('Write results to file: {}'.format(self.output_file))
        output = pd.DataFrame(data={'review': self.clean_test, 'score_predict':self.result, 'score':self.test['score']})

        score = 0
        for i in range(0, len(self.result)):
            if int(self.result[i]) >= 6 and int(self.test['score'][i] >= 6):
                score += 1
            elif int(self.result[i]) < 5 and int(self.test['score'][i] < 5):
                score += 1
        
        self.logger.info('Size of training: {}'.format(len(self.clean_train)))
        self.logger.info('Size of test: {}'.format(len(self.clean_test)))
        self.logger.info('Predicted correct: {}/{} {:3.2f}%'.format(score, len(self.clean_test), score/len(self.clean_test)*100))

        output.to_csv(self.output_file, index=False, quoting=3)


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    c = Crawler('input.txt', 'output_train.csv', 'output_test.csv')
    b = BOW(stop_words_file='stopwords2.txt',
            file_name_train='output_train.csv',
            file_name_test='output_test.csv',
            output_file='output.csv')
