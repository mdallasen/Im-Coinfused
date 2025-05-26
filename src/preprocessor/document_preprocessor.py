import pandas as pd
import spacy
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

class Preprocessor:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent
        self.data_dir = base_dir.parent.parent / "data"  
        self.wiki_path = self.data_dir / "crypto_wiki_corpus.jsonl"
        self.nlp = spacy.load("en_core_web_sm")
        self.df = None

        model_name = "valhalla/t5-base-qg-hl"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

        self.qg_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    def load_data(self):
        wiki_df = pd.read_json(self.wiki_path, lines=True)
        wiki_df['source'] = 'wiki'
        self.df = wiki_df

        # Uncomment if you want to load Reddit data as well
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
    
    def chunk_text(self, text, max_words=150):
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i+max_words])
            chunks.append(chunk)
        return chunks

    def summarize_chunks(self, text):
        chunks = self.chunk_text(text)
        summaries = []
        for chunk in chunks:
            try:
                summary = self.summarizer(chunk, max_length=50, min_length=10, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                print(f"Summarization error: {e}")
                summaries.append(chunk)
        combined_summary = " ".join(summaries)
        return combined_summary

    def generate_questions(self, text):
        # Summarize the text with chunking
        summary = self.summarize_chunks(text)

        # Truncate summary tokens to avoid exceeding model limit
        tokens = self.qg_pipeline.tokenizer.tokenize(summary)
        if len(tokens) > 200:
            tokens = tokens[:200]
            summary = self.qg_pipeline.tokenizer.convert_tokens_to_string(tokens)

        # Generate questions from the summary
        input_text = f"generate question: {summary}"
        try:
            outputs = self.qg_pipeline(
                input_text,
                max_length=64,
                num_return_sequences=3,
                truncation=True 
            )
            return [out['generated_text'] for out in outputs]
        except Exception as e:
            print(f"QG error: {e}")
            return []

    def process(self):
        self.df['content'] = self.df['content'].apply(self.normalize_text)
        self.df['entities'] = self.df['content'].apply(self.extract_entities)

        # Expand entities into separate columns
        df_entities = self.df['entities'].apply(lambda x: pd.Series(x))
        df_entities.columns = [f"entity_{i+1}" for i in range(df_entities.shape[1])]
        self.df = pd.concat([self.df.drop(columns=['entities']), df_entities], axis=1)

        # Use the new generate_questions method that summarizes + chunks
        self.df['questions'] = self.df['content'].apply(self.generate_questions)

        return self.df
    
    def save(self, df, path):
        print(f"Saving data to {path}")
        df.to_json(path, orient="records", lines=True, force_ascii=False)

    def run(self, processed_save_path=None):
        processed_save_path = Path(processed_save_path or self.data_dir / "crypto_corpus.jsonl")
        self.load_data()
        df = self.process() 
        self.save(df, processed_save_path)
        return df

if __name__ == "__main__":
    preprocessor = Preprocessor()
    df = preprocessor.run()