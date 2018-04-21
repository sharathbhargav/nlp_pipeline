from praw import Reddit


def get_reddit_instance():
    r = Reddit(client_id='TvDxclK5OdPpZg',
                    client_secret='J3xtU_M7cDvxuC3LcutVG8oSM6k',
                    user_agent='trial by /u/eighthsemproject')
    return r