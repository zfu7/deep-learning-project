import numpy as np
import twitter, re, datetime, pandas as pd

#TweetMiner function from Mike Roman

class TweetMiner(object):

    
    def __init__(self, api, result_limit = 20):
        
        self.api = api        
        self.result_limit = result_limit
        

    def mine_user_tweets(self, user="HillaryClinton", mine_retweets=False, max_pages=20):

        data           =  []
        last_tweet_id  =  False
        page           =  1
        
        while page <= max_pages:
            
            if last_tweet_id:
                statuses   =   self.api.GetUserTimeline(screen_name=user, count=self.result_limit, max_id=last_tweet_id - 1, include_rts=mine_retweets)
                statuses = [ _.AsDict() for _ in statuses]
            else:
                statuses   =   self.api.GetUserTimeline(screen_name=user, count=self.result_limit, include_rts=mine_retweets)
                statuses = [_.AsDict() for _ in statuses]
                
            for item in statuses:
                # Using try except here.
                # When retweets = 0 we get an error (GetUserTimeline fails to create a key, 'retweet_count')
                try:
                    mined = {
                        'tweet_id':        item['id'],
                        'handle':          item['user']['screen_name'],
                        'retweet_count':   item['retweet_count'],
                        'text':            item['full_text'],
                        'mined_at':        datetime.datetime.now(),
                        'created_at':      item['created_at'],
                    }
                
                except:
                        mined = {
                        'tweet_id':        item['id'],
                        'handle':          item['user']['screen_name'],
                        'retweet_count':   0,
                        'text':            item['full_text'],
                        'mined_at':        datetime.datetime.now(),
                        'created_at':      item['created_at'],
                    }
                
                last_tweet_id = item['id']
                data.append(mined)
                
            page += 1
            
        return data

twitter_keys = {
    'consumer_key':        'L4sziHBqV4VUIfKezbos0JMVl',
    'consumer_secret':     'lJau6R7GIHFwoGR5wB3PlLQPXBChwzJFJ9WGXXtazcDSA1Vb1X',
    'access_token_key':    '941359629606539264-05XcmQfdwMXTbPNWS3r7cZThvbQBxCK',
    'access_token_secret': 'VdE3VJVk6oxbohQGcw7WYA5Tg4Sr8kW9duTO1wxmB6qXk'
}

api = twitter.Api(
    consumer_key         =   twitter_keys['consumer_key'],
    consumer_secret      =   twitter_keys['consumer_secret'],
    access_token_key     =   twitter_keys['access_token_key'],
    access_token_secret  =   twitter_keys['access_token_secret'],
    tweet_mode = 'extended'
)



miner = TweetMiner(api, result_limit=200)
hillary = miner.mine_user_tweets(user="HillaryClinton")
donald = miner.mine_user_tweets(user="realDonaldTrump")
hillary = pd.DataFrame(hillary)
donald = pd.DataFrame(donald)
# print(hillary)
hillary.to_csv('../data/mining_hillary.csv')
donald.to_csv('../data/mining_donald.csv')
print ('Done.')
