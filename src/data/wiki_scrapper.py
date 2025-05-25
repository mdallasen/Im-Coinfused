import wikipedia
import json
from pathlib import Path
from crypto_topics import topics

class WikiScapper:
    def __init__(self, titles):
        self.titles = titles
        self.articles = {}

    def fetch_articles(self):
        for title in self.titles:
            page = wikipedia.page(title)
            self.articles[title] = page.content

    def save_to_jsonl(self, filename):
        base_dir = Path(__file__).resolve().parent
        data_dir = base_dir.parent.parent / "data"
        filepath = data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for title, content in self.articles.items():
                record = {"title": title, "content": content}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def extract(cls):
        collector = cls(topics)
        collector.fetch_articles()
        collector.save_to_jsonl("crypto_wiki_corpus.jsonl")

if __name__ == "__main__":
    WikiScapper.extract()
