def process_comments(objects, fh):
    for comment in objects:
        if type(comment).__name__ == 'Comment':
            fh.write(str(comment.body) + '\n')
        elif type(comment).__name__ == 'MoreComments':
            process_comments(comment.comments(), fh)


'''
# To fetch every comment
def process_comments(objects, fh):
    comment_body = str()
    for comment in objects:
        if type(comment).__name__ == 'Comment':
            fh.write(str(comment.body) + '\n')
            process_comments(comment.replies, fh)
        elif type(comment).__name__ == 'MoreComments':
            process_comments(comment.comments(), fh)
'''