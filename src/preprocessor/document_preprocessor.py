import pandas as pd 
import spacy
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from pathlib import Path
class Preprocessor:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent

        self.data_dir = base_dir / "data"
        self.wiki_path = self.data_dir / "wiki.csv"
        self.reddit_path = self.data_dir / "reddit.csv"
        self.nlp = spacy.load("en_core_web_sm")
        self.df = None

    def load_data(self): 
        wiki_df = pd.read_json(self.wiki_path)
        wiki_df['source'] = 'wiki'
        wiki_df.dropna(subset=["url"], inplace=True)
        wiki_df.rename(columns={'title': 'title', 'content': 'content'}, inplace=True)

        # reddit_df = pd.read_csv(self.reddit_path)
        # reddit_df['source'] = 'reddit'
        # reddit_df.dropna(subset=["post_url"], inplace=True)
        # reddit_df.rename(columns={'post_title': 'title', 'body': 'content'}, inplace=True)

        # self.df = pd.concat([wiki_df, reddit_df], ignore_index=True)

    def normalize_text(self, text):
        text = text.lower()
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return ' '.join(sentences)
    
    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'GPE', 'PRODUCT', 'PERSON', 'MONEY', 'DATE', 'EVENT', 'LAW', 'QUANTITY']:
                entities.append(ent.text)
        return list(set(entities))  
    
    def process(self):
        self.df['content'] = self.df['content'].apply(self.normalize_text)
        self.df['entities'] = self.df['content'].apply(self.extract_entities)
        df_entities = self.df['entities'].apply(lambda x: pd.Series(x))
        df_entities.columns = [f"entity_{i+1}" for i in range(df_entities.shape[1])]
        self.df = pd.concat([self.df.drop(columns=['entities']), df_entities], axis=1)

    def generate_questions(self):
        model_name = "valhalla/t5-base-qg-hl"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

        qa_pairs = []

        for idx, row in self.df.iterrows():
            text = row['content']
            entities = self.extract_entities(text)

            for answer in entities:
                if answer in text:
                    highlighted_text = text.replace(answer, f"<hl> {answer} <hl>", 1)
                else:
                    continue  

                input_text = f"generate question: {highlighted_text}"
                inputs = tokenizer(input_text, return_tensors="tf", truncation=True)

                outputs = model.generate(**inputs, max_length=64)
                question = tokenizer.decode(outputs[0], skip_special_tokens=True)

                qa_pairs.append({
                    "context": text,
                    "answer": answer,
                    "question": question,
                    "source": row['source'],
                    "title": row['title']
                })

        return pd.DataFrame(qa_pairs)
    
    def save(self, path):
        self.df.to_csv(path, index=False, encoding='utf-8')

    def run(self, save_path="data/crypto_corpus.csv", qg_save_path="data/crypto_qa.csv"):
        self.load_data()
        self.process()
        self.save(save_path)

        qa_df = self.generate_questions()
        qa_df.to_csv(qg_save_path, index=False, encoding='utf-8')

        return self.df, qa_df
    
if __name__ == "__main__":
    preprocessor = Preprocessor()
    df, qa_df = preprocessor.run()