import wikipedia
import json
from pathlib import Path
from data.crypto_topics import topics

class WikiScapper:
    def __init__(self, titles):
        self.titles = titles
        self.articles = {}

    def fetch_articles(self):
        print(f"Starting to fetch {len(self.titles)} articles...")
        for idx, title in enumerate(self.titles, 1):
            print(f"[{idx}/{len(self.titles)}] Fetching article: '{title}'")
            page = wikipedia.page(title)
            self.articles[title] = page.content
        print("Finished fetching all articles.")

    def save_to_jsonl(self, filename):
        base_dir = Path(__file__).resolve().parent
        data_dir = base_dir.parent.parent / "data"
        filepath = data_dir / filename

        print(f"Saving articles to {filepath} ...")
        with open(filepath, 'w', encoding='utf-8') as f:
            for idx, (title, content) in enumerate(self.articles.items(), 1):
                record = {"title": title, "content": content}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if idx % 10 == 0 or idx == len(self.articles):
                    print(f"  Saved {idx}/{len(self.articles)} articles...")
        print("All articles saved successfully.")

    @classmethod
    def extract(cls):
        print("Starting extraction process...")
        collector = cls(topics)
        collector.fetch_articles()
        collector.save_to_jsonl("crypto_wiki_corpus.jsonl")
        print("Extraction process completed.")
