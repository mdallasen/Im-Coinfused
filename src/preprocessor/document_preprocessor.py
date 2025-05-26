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

        print("Loading summarization model...")
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )
        print("Summarization model loaded.")

        model_name = "valhalla/t5-base-qg-hl"
        print(f"Loading question generation model: {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.qg_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer
        )
        print("Question generation model loaded.")

    def load_data(self):
        print(f"Loading data from {self.wiki_path} ...")
        self.df = pd.read_json(self.wiki_path, lines=True)
        self.df['source'] = 'wiki'
        print(f"Loaded {len(self.df)} records.")

    def normalize_text(self, text):
        print(f"Normalizing text (length: {len(text)} chars)...")
        text = text.lower()
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return ' '.join(sentences)
    
    def extract_entities(self, text):
        print(f"Extracting entities (length: {len(text)} chars)...")
        doc = self.nlp(text)
        entities = [
            ent.text for ent in doc.ents 
            if ent.label_ in ['ORG', 'GPE', 'PRODUCT', 'PERSON', 'MONEY', 'DATE', 'EVENT', 'LAW', 'QUANTITY']
        ]
        unique_entities = list(set(entities))
        print(f"Found {len(unique_entities)} unique entities.")
        return unique_entities
    
    def chunk_text(self, text, max_words= 500):
        words = text.split()
        print(f"Chunking text of {len(words)} words into chunks of max {max_words} words each...")
        return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

    def summarize_chunks(self, text):
        chunks = self.chunk_text(text)
        print(f"Summarizing {len(chunks)} chunks...")
        summaries = []
        for i, chunk in enumerate(chunks, 1):
            print(f"  Summarizing chunk {i}/{len(chunks)} (length: {len(chunk.split())} words)...")
            try:
                summary = self.summarizer(chunk, max_length=50, min_length=10, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                print(f"    Warning: summarization failed on chunk {i}: {e}")
                summaries.append(chunk)

        combined = " ".join(summaries)
        print("Generating combined summary of all chunks...")
        try:
            short_summary = self.summarizer(combined, max_length=60, min_length=15, do_sample=False)
            print(f"Combined summary length: {len(short_summary[0]['summary_text'].split())} words.")
            return short_summary[0]['summary_text']
        except Exception as e:
            print(f"Warning: meta-summarization failed: {e}")
            return combined

    def generate_questions(self, text):
        print("Generating questions from summarized text...")
        summary = self.summarize_chunks(text)
        print(f"Summary for question generation: {summary[:100]}...")  
        try:
            encoded = self.qg_pipeline.tokenizer.encode(summary, truncation=True, max_length=512)
            truncated_summary = self.qg_pipeline.tokenizer.decode(encoded, skip_special_tokens=True)
            input_text = f"generate question: {truncated_summary}"
            outputs = self.qg_pipeline(
                input_text,
                max_new_tokens=64,
                num_return_sequences=3,
                truncation=True,
                return_full_text=False
            )
            questions = [out['generated_text'] for out in outputs]
            print(f"Generated {len(questions)} questions.")
            return questions
        except Exception as e:
            print(f"Warning: question generation failed: {e}")
            return []

    def process(self):
        print("Starting preprocessing of content...")
        self.df['content'] = self.df['content'].apply(self.normalize_text)
        self.df['entities'] = self.df['content'].apply(self.extract_entities)

        df_entities = self.df['entities'].apply(lambda x: pd.Series(x))
        if not df_entities.empty:
            df_entities.columns = [f"entity_{i + 1}" for i in range(df_entities.shape[1])]
            self.df = pd.concat([self.df.drop(columns=['entities']), df_entities], axis=1)
            print(f"Extracted entities expanded into {df_entities.shape[1]} columns.")

        self.df['questions'] = self.df['content'].apply(self.generate_questions)
        print("Preprocessing complete.")
        return self.df
    
    def save(self, df, path):
        print(f"Saving processed data to {path} ...")
        df.to_json(path, orient="records", lines=True, force_ascii=False)
        print("Data saved successfully.")

    def run(self, processed_save_path=None):
        processed_save_path = Path(processed_save_path or self.data_dir / "crypto_corpus.jsonl")
        self.load_data()
        df = self.process() 
        self.save(df, processed_save_path)
        return df
