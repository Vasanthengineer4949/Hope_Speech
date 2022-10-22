import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

class Embedder:

    def __init__(self, data_path, lang, model_ckpt, train_or_test_data):
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)
        self.lang = lang
        self.model_ckpt = model_ckpt
        self.train_or_test_data = train_or_test_data
    
    def embedding_generator(self, df, model):
        tqdm.pandas()
        model = SentenceTransformer(model)
        df['Embeddings'] = df['text'].progress_apply(lambda x: model.encode(x))
        return df

    def inputgen(self, df):
        a = []
        for i in tqdm(range(len(df))):
            a.append(np.array(df["Embeddings"][i]))
        a = np.array(a)
        print(a.shape)
        return a

    def run(self):
        inputgen_input = self.embedding_generator(self.df, self.model_ckpt)
        print(f"{self.lang}_{self.train_or_test_data}_Embedding completed")
        embed_output = self.inputgen(inputgen_input)
        print(f"{self.lang}_{self.train_or_test_data}_Array completed")
        embed = pd.DataFrame(embed_output)
        df = pd.concat([self.df, embed], axis=1)
        print(f"{self.lang}_{self.train_or_test_data}_Dataframe completed")
        df = df.drop(["Embeddings", "text"], axis=1)
        df.to_csv(f"artifacts/embeddings/{self.lang}_{self.train_or_test_data}_embeddings.csv", index=False)
        print(f"{self.lang}_{self.train_or_test_data}_Dataframe saved as csv")
        return df





