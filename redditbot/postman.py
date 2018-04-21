from redditbot.reddit_auth import get_reddit_instance
from redditbot.commentator import process_comments
from redditbot.util import get_file_name, file_location
import datetime


def get_hot_posts(num_posts=100):
    reddit = get_reddit_instance()
    directory = file_location + 'hot/'
    for post in reddit.front.hot(limit=num_posts):
        fname = directory + datetime.datetime.now().strftime("%y-%m-%d") + '_' + get_file_name(post.title)
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
    for post in reddit.front.controversial(limit=num_posts):
        fname = directory + datetime.datetime.now().strftime("%y-%m-%d") + '_' + get_file_name(post.title)
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
    for post in reddit.front.gilded(limit=num_posts):
        fname = directory + datetime.datetime.now().strftime("%y-%m-%d") + '_' + get_file_name(post.title)
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
    for post in reddit.front.new(limit=num_posts):
        fname = directory + datetime.datetime.now().strftime("%y-%m-%d") + '_' + get_file_name(post.title)
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
    for post in reddit.front.rising(limit=num_posts):
        fname = directory + datetime.datetime.now().strftime("%y-%m-%d") + '_' + get_file_name(post.title)
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
    for post in reddit.front.top(limit=num_posts):
        fname = directory + datetime.datetime.now().strftime("%y-%m-%d") + '_' + get_file_name(post.title)
        print(fname)
        f = open(fname, 'w+')
        f.write(post.title + '\n')
        f.write(post.selftext + '\n')
        process_comments(post.comments.list(), f)
        f.close()
    return directory


#get_controversial_posts(5)