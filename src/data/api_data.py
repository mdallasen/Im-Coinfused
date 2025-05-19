import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import os

class RedditScraper:
    def __init__(self, client_id, client_secret, username, password, user_agent="MyAPI/0.0.1"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.user_agent = user_agent
        self.base_url = "https://oauth.reddit.com"
        self.token = self.get_token()

    def get_token(self):
        auth_url = "https://www.reddit.com/api/v1/access_token"
        auth = HTTPBasicAuth(self.client_id, self.client_secret)

        data = {
            'grant_type': 'password',
            'username': self.username,
            'password': self.password
        }

        headers = {
            'User-Agent': self.user_agent
        }

        res = requests.post(auth_url, auth=auth, headers=headers, data=data)

        if res.status_code != 200:
            raise Exception(f"Failed to get access token: {res.text}")

        return res.json().get("access_token")

    def get_headers(self):
        return {
            'Authorization': f'bearer {self.token}',
            'User-Agent': self.user_agent
        }

    def get_top_threads(self, subreddit, time_filter="all", limit = 3):
        url = f"{self.base_url}/r/{subreddit}/top"
        params = {
            "t": time_filter,
            "limit": limit
        }

        res = requests.get(url, headers=self.get_headers(), params=params)

        if res.status_code != 200:
            raise Exception(f"Failed to fetch top threads: {res.text}")

        posts = []
        for post in res.json()['data']['children']:
            post_data = post['data']
            posts.append({
                'post_id': post_data['id'],
                'title': post_data['title'],
                'url': f"https://www.reddit.com{post_data['permalink']}",
                'score': post_data['score'],
                'num_comments': post_data['num_comments'],
                'author': post_data['author']
            })

        return pd.DataFrame(posts)

    def get_top_comments(self, post_id, limit=10):
        url = f"{self.base_url}/comments/{post_id}.json"
        params = {
            "limit": limit,
            "sort": "top"
        }

        res = requests.get(url, headers=self.get_headers(), params=params)

        if res.status_code != 200:
            raise Exception(f"Failed to fetch comments for post {post_id}: {res.text}")

        comments = []
        for comment in res.json()[1]['data']['children']:
            if comment['kind'] == 't1':  # Make sure it's a comment, not meta info
                comment_data = comment['data']
                comments.append({
                    'body': comment_data['body'],
                })

        return pd.DataFrame(comments)

    def get_top_threads_and_comments(self, subreddit, post_limit, comment_limit):
        threads_df = self.get_top_threads(subreddit, limit=post_limit)

        all_comments = []
        for index, row in threads_df.iterrows():
            post_id = row['post_id']
            comments_df = self.get_top_comments(post_id, limit=comment_limit)

            # Add the post title and URL for context
            comments_df['post_title'] = row['title']
            comments_df['post_url'] = row['url']
            
            all_comments.append(comments_df)

        # Combine all the comments into a single DataFrame
        full_df = pd.concat(all_comments, ignore_index=True)
        return full_df

# Example usage
scraper = RedditScraper(
    client_id="6iuJ23SHRkb1Jm6z9_unpA",
    client_secret="jlIMhe1LFWNS7QeADXOVPwktaHDCwA",
    username="Flaky_Soil_3691",
    password="Matthew123@"
)

df = scraper.get_top_threads_and_comments("cryptocurrency", post_limit=1000, comment_limit=3)
print(df.head())
# folder_path = os.path.expanduser(r"~/src/data")
# file_path = os.path.join(folder_path, "top_crypto_threads.csv")
# df.to_csv(file_path, index=False)
# print(f"Data saved to {file_path}")