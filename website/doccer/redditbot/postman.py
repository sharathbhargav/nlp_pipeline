from .reddit_auth import get_reddit_instance
from .commentator import process_comments
from .util import get_file_name, file_location
import datetime


def get_hot_posts(num_posts=100):
    reddit = get_reddit_instance()
    directory = file_location + 'hot/'
    i = 0
    for post in reddit.front.hot(limit=num_posts):
        if type(post).__name__ == 'Submission':
            fname = directory + datetime.datetime.now().strftime("%y-%m-%d") + '_' + str(i) #get_file_name(post.title)
            i += 1
            print(fname)
            f = open(fname, 'w')
            f.write(post.title + '\n')
            f.write(post.selftext + '\n')
            process_comments(post.comments.list(), f)
            f.close()
    return directory


def get_controversial_posts(num_posts=100):
    reddit = get_reddit_instance()
    directory = file_location + 'controversial/'
    i = 0
    for post in reddit.front.controversial(limit=num_posts):
        if type(post).__name__ == 'Submission':
            fname = directory + datetime.datetime.now().strftime("%y-%m-%d") + '_' + str(i) #get_file_name(post.title)
            i += 1
            print(fname)
            f = open(fname, 'w+')
            f.write(post.title + '\n')
            f.write(post.selftext + '\n')
            process_comments(post.comments.list(), f)
            f.close()
    return directory


def get_gilded_posts(num_posts=100):
    reddit = get_reddit_instance()
    directory = file_location + 'gilded/'
    i = 0
    for post in reddit.front.gilded(limit=num_posts):
        if type(post).__name__ == 'Submission':
            fname = directory + datetime.datetime.now().strftime("%y-%m-%d") + '_' + str(i) #get_file_name(post.title)
            i += 1
            print(fname)
            f = open(fname, 'w+')
            f.write(post.title + '\n')
            f.write(post.selftext + '\n')
            process_comments(post.comments.list(), f)
            f.close()
    return directory


def get_new_posts(num_posts=100):
    reddit = get_reddit_instance()
    directory = file_location + 'new/'
    i = 0
    for post in reddit.front.new(limit=num_posts):
        if type(post).__name__ == 'Submission':
            fname = directory + datetime.datetime.now().strftime("%y-%m-%d") + '_' + str(i) #get_file_name(post.title)
            i += 1
            print(fname)
            f = open(fname, 'w+')
            f.write(post.title + '\n')
            f.write(post.selftext + '\n')
            process_comments(post.comments.list(), f)
            f.close()
    return directory


def get_rising_posts(num_posts=100):
    reddit = get_reddit_instance()
    directory = file_location + 'rising/'
    i = 0
    for post in reddit.front.rising(limit=num_posts):
        if type(post).__name__ == 'Submission':
            fname = directory + datetime.datetime.now().strftime("%y-%m-%d") + '_' + str(i) #get_file_name(post.title)
            i += 1
            print(fname)
            f = open(fname, 'w+')
            f.write(post.title + '\n')
            f.write(post.selftext + '\n')
            process_comments(post.comments.list(), f)
            f.close()
    return directory


def get_top_posts(num_posts=100):
    reddit = get_reddit_instance()
    directory = file_location + 'top/'
    i = 0
    for post in reddit.front.top(limit=num_posts):
        if type(post).__name__ == 'Submission':
            fname = directory + datetime.datetime.now().strftime("%y-%m-%d") + '_' + str(i) #get_file_name(post.title)
            i += 1
            print(fname)
            f = open(fname, 'w+')
            f.write(post.title + '\n')
            f.write(post.selftext + '\n')
            process_comments(post.comments.list(), f)
            f.close()
    return directory