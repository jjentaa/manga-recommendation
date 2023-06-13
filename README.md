# Manga-recommendation
It is a recommendation model for recommend mangas by using query(user requirement).\

## Get start
First, you have to pip install the necessary library.
```
!pip install sentence-transformers==2.2.2
```

or you can create `requirements.txt` and in this file put :
```
sentence-transformers==2.2.2
```

## Finetune
I just generate query by using [Alpaca-LoRA](https://colab.research.google.com/drive/1eWAmesrW99p7e1nah5bipn0zikMb8XYC?usp=sharing). and finetune it.

This is a code for generate querys. 

I provided 2 scripts of finetuning code.
```
--fine_tune_eng.ipynb
--fine_tune_multi.ipynb
```
The first script used `"all-MiniLM-L6-v"` for being a pretrain model.\
The second script used `"paraphrase-multilingual-mpnet-base-v2"` for being a pretrain model.

## Usage
There are 2 finetuned models for encode the word to vector.
1. Madnesss/fine-tune-all-MiniLM-L6-v2 : It is for English only.
2. Madnesss/fine-tune-paraphrase-multilingual-mpnet-base-v2 : It is multilingual model.

### Finding cosine similarity score
```
from sentence_transformers import SentenceTransformer, util

path = "Madnesss/fine-tune-all-MiniLM-L6-v2" # or path = "Madnesss/fine-tune-paraphrase-multilingual-mpnet-base-v2"
model = SentenceTransformer(path)

#encode
embedding1 = model.encode(["example sentence1", "example sentence2", "example sentence3"], convert_to_tensor=True)
embedding2 = model.encode(["example sentence4", "example sentence5", "example sentence6"], convert_to_tensor=True)

#compute cosine sim scores
cosine_scores = util.cos_sim(embeddings1, embeddings2)

#Output the scores
print(cosine_scores) #tensor([0.1, 0,2, 0.3, ....])
```

## Blog
I have written a blog for giving more details. You can find out [here](https://medium.com/@madness_/manga-recommendation-️-a8147d933c51).
