import requests
import os
import bs4
import logging
import csv
import re
import configparser
import pandas as pd
import numpy as np
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from gensim.models import KeyedVectors

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
        
        if os.path.exists(self.output_file_test) and os.path.exists(self.output_file_train):
            outfile_test.to_csv(self.output_file_test, header=False, quoting=3, index_label='id', sep='\t', mode='a')
            outfile_train.to_csv(self.output_file_train, header=False, quoting=3, index_label='id', sep='\t', mode='a')
        else:
            outfile_test.to_csv(self.output_file_test, quoting=3, index_label='id', sep='\t', mode='a')
            outfile_train.to_csv(self.output_file_train, quoting=3, index_label='id', sep='\t', mode='a')

class GetWordVectors:
    def __init__(self, stop_words_file, file_name_train, file_name_test, output_file, word2vec_model_file, binary_w2v_format, num_features=300, mode='w2v'):
        self.logger = logging.getLogger('main.' + self.__class__.__name__)

        self.output_file = output_file
        self.num_features = num_features
        self.mode = mode

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
        
        self.result_bow = None
        self.result_w2v = None

        if self.mode == 'bow' or self.mode == 'both':
            # run classification
            train_feature_vecs = self.get_feature_vec()
            self.classify(train_feature_vecs)

            # run tests
            self.result_bow = self.run_test_bow()

        elif self.mode == 'w2v' or self.mode == 'both':
            # load W2V model
            self.load_word_2_vec(word2vec_model_file, binary_w2v_format)
            
            # in order of further processing we need vectors of words instead of sentences
            clean_train_list = []
            clean_test_list = []
            for c in self.clean_train:
                clean_train_list.append(list(c.split(' ')))
            for c in self.clean_test:
                clean_test_list.append(list(c.split(' ')))

            train_feature_vecs = self.get_avg_feature_vec(clean_train_list)
            test_feature_vecs = self.get_avg_feature_vec(clean_test_list)

            self.classify(train_feature_vecs)

            self.result_w2v = self.run_test_w2v(test_feature_vecs)
        else:
            self.logger.critical('Mode not recognized: {}'.format(self.mode))
            exit(1)
            
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

    def load_word_2_vec(self, word2vec_model_file, binary_w2v_format):
        self.logger.info('Word 2 vec model loading: {}'.format(word2vec_model_file))
        self.model = KeyedVectors.load_word2vec_format(word2vec_model_file, binary=binary_w2v_format, unicode_errors='ignore')

    def make_feature_vec(self, words):
        self.logger.debug('Make feature vectors')
        feature_vec = np.zeros((self.num_features,), dtype='float32')

        nwords = 0

        # index to word set
        i2w_set = set(self.model.index2word)

        for word in words:
            if word in i2w_set:
                nwords += 1
                feature_vec = np.add(feature_vec, self.model[word])

        if nwords:
            feature_vec = np.divide(feature_vec, nwords)

        return feature_vec

    def get_avg_feature_vec(self, clean_reviews):
        self.logger.info('Get avarage feature vector')
        counter = 0

        review_feature_vec = np.zeros((len(clean_reviews), self.num_features), dtype='float32')

        review_len = len(clean_reviews)
        review_len_10percent = int(review_len/10)
        for review in clean_reviews:
            if not counter % review_len_10percent:
                self.logger.info('{:3.0f}% done!'.format(counter/review_len * 100))
            review_feature_vec[counter] = self.make_feature_vec(review)
            counter += 1
        self.logger.info('100% done!')

        return review_feature_vec

    def get_feature_vec(self):
        self.logger.info('Create count vectorizer')
        self.vectorizer = CountVectorizer(analyzer='word',
                                          tokenizer=None,
                                          preprocessor=None,
                                          stop_words=None,
                                          max_features=self.num_features)

        self.logger.info('Fit transform')
        train_data_features = self.vectorizer.fit_transform(self.clean_train)

        self.logger.info('To array..')
        train_data_features = train_data_features.toarray()

        return train_data_features

    def classify(self, train_data_vec):
        self.logger.info('Create classifier')
#        self.forest = RandomForestClassifier(n_estimators=100)
        self.regression = LogisticRegression(n_jobs=8)

        self.logger.info('Fit classifier')
#        self.forest = self.forest.fit(train_data_vec, self.train['score'])
        self.regression = self.regression.fit(train_data_vec, self.train['score'])
        
    def run_test_bow(self):
        self.logger.info('Run tests...')
        test_data_features = self.vectorizer.transform(self.clean_test)

        self.logger.info('To array...')
        test_data_features = test_data_features.toarray()

        self.logger.info('Predict...')
        result = self.regression.predict(test_data_features)

        return result

    def run_test_w2v(self, test_feature_vecs):
        self.logger.info('Predict...')
        result = self.regression.predict(test_feature_vecs)

        return result

    def write_results(self):
        self.logger.info('Write results to file: {}'.format(self.output_file))
        if self.mode == 'bow':
            data = {'review': self.clean_test, 'score_predict_bow':self.result_bow, 'score':self.test['score']}
        elif self.mode == 'w2v':
            data = {'review': self.clean_test, 'score_predict_w2v':self.result_w2v, 'score':self.test['score']}
        elif self.mode == 'both':
            data = {'review': self.clean_test, 'score_predict_bow':self.result_bow, 'score_predict_w2v': self.result_w2v, 'score':self.test['score']}

        output = pd.DataFrame(data)

        self.logger.info('Size of training: {}'.format(len(self.clean_train)))
        self.logger.info('Size of test: {}'.format(len(self.clean_test)))

        if self.mode == 'bow' or self.mode == 'both':
            score = 0
            for i in range(0, len(self.result_bow)):
                if int(self.result_bow[i]) >= 6 and int(self.test['score'][i] >= 6):
                    score += 1
                elif int(self.result_bow[i]) < 6 and int(self.test['score'][i] < 6):
                    score += 1
        
            self.logger.info('Predicted bow correct: {}/{} {:3.2f}%'.format(score, len(self.clean_test), score/len(self.clean_test)*100))
        if self.mode == 'w2v' or self.mode == 'both':
            score = 0
            for i in range(0, len(self.result_w2v)):
                if int(self.result_w2v[i]) >= 6 and int(self.test['score'][i] >= 6):
                    score += 1
                elif int(self.result_w2v[i]) < 6 and int(self.test['score'][i] < 6):
                    score += 1

            self.logger.info('Predicted w2v correct: {}/{} {:3.2f}%'.format(score, len(self.clean_test), score/len(self.clean_test)*100))

        output.to_csv(self.output_file, index=False, quoting=3)

if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    
    config = configparser.ConfigParser()
    config.read('config.ini')

    c = Crawler(config.get('FilmwebCrawler', 'input_file'),
                config.get('FilmwebCrawler', 'output_file_train'),
                config.get('FilmwebCrawler', 'output_file_test'))
    b = GetWordVectors(stop_words_file=config.get('BOW', 'stop_words_file'),
                       file_name_train=config.get('BOW', 'file_name_train'),
                       file_name_test=config.get('BOW', 'file_name_test'),
                       output_file=config.get('BOW', 'output_file'),
                       word2vec_model_file=config.get('BOW', 'w2v_model_file'),
                       binary_w2v_format=config.get('BOW', 'binary_w2v_format'),
                       num_features=int(config.get('BOW', 'num_features')),
                       mode=config.get('BOW', 'mode'))

