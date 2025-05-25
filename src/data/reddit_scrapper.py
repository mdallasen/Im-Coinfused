import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import json
from pathlib import Path

class RedditScraper:
    def __init__(self):
        self.client_id = "6iuJ23SHRkb1Jm6z9_unpA"
        self.client_secret = "jlIMhe1LFWNS7QeADXOVPwktaHDCwA"
        self.username = "Flaky_Soil_3691"
        self.password = "Matthew123@"
        self.user_agent = "MyAPI/0.0.1"
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
        headers = {'User-Agent': self.user_agent}
        res = requests.post(auth_url, auth=auth, headers=headers, data=data)
        if res.status_code != 200:
            raise Exception(f"Failed to get access token: {res.text}")
        return res.json().get("access_token")

    def get_headers(self):
        return {
            'Authorization': f'bearer {self.token}',
            'User-Agent': self.user_agent
        }

    def get_top_threads(self, subreddit, time_filter="all", limit=3):
        url = f"{self.base_url}/r/{subreddit}/top"
        params = {"t": time_filter, "limit": limit}
        res = requests.get(url, headers=self.get_headers(), params=params)

        if res.status_code != 200:
            raise Exception(f"Failed to fetch top threads: {res.status_code} {res.text}")

        try:
            json_data = res.json()
            children = json_data['data']['children']
        except (KeyError, json.JSONDecodeError) as e:
            raise Exception(f"Unexpected response structure or JSON error: {e} - Response: {res.text}")

        posts = []
        for post in children:
            post_data = post['data']
            posts.append({
                'post_id': post_data['id'],
                'title': post_data['title'],
                'url': f"https://www.reddit.com{post_data['permalink']}",
            })
        return pd.DataFrame(posts)

    def get_top_comments(self, post_id, limit=10):
        url = f"{self.base_url}/comments/{post_id}.json"
        params = {"limit": limit, "sort": "top"}
        res = requests.get(url, headers=self.get_headers(), params=params)
        if res.status_code != 200:
            raise Exception(f"Failed to fetch comments for post {post_id}: {res.text}")
        comments = []
        for comment in res.json()[1]['data']['children']:
            if comment['kind'] == 't1':
                comment_data = comment['data']
                comments.append({'body': comment_data['body']})
        return pd.DataFrame(comments)

    def get_top_threads_and_comments(self, subreddit, post_limit=3, comment_limit=10):
        threads_df = self.get_top_threads(subreddit, limit=post_limit)
        all_comments = []
        for _, row in threads_df.iterrows():
            post_id = row['post_id']
            comments_df = self.get_top_comments(post_id, limit=comment_limit)
            comments_df['post_title'] = row['title']
            comments_df['post_url'] = row['url']
            all_comments.append(comments_df)
        full_df = pd.concat(all_comments, ignore_index=True)
        return full_df

    def save_to_jsonl(self, df, filename):
        base_dir = Path(__file__).resolve().parent
        data_dir = base_dir.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        filepath = data_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            for record in df.to_dict(orient='records'):
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"Saved Reddit data to {filepath}")

    @classmethod
    def extract(cls):
        scraper = cls() 
        df = scraper.get_top_threads_and_comments(subreddit="CryptoCurrency", post_limit=10, comment_limit=3)
        scraper.save_to_jsonl(df, "reddit_crypto_comments.jsonl")

if __name__ == "__main__":
    RedditScraper.extract()
