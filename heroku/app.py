import pandas as pd
import numpy as np
import streamlit as st

st.sidebar.title("LyricsMatch")

st.sidebar.info("This is a POC application for LyricsMatch.")

english_lyrics = pd.read_csv("english_lyrics_first50.csv")
english_lyrics = english_lyrics[english_lyrics['lang'] == 'en']

hindi_lyrics = pd.read_csv("experiments/data/hindilyrics_pratik.csv")

import numpy as np
dim = 1024

hindi_embeddings = np.fromfile("experiments/data/embeddings/pratik_hindi_embeddings.raw", dtype=np.float32, count=-1)
hindi_embeddings.resize(hindi_embeddings.shape[0] // dim, dim)

english_embeddings = np.load("english_embeddings.npy")
# english_embeddings.resize(english_embeddings.shape[0] // dim, dim)

d=1024
import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index
# print(index.is_trained)
index.add(hindi_embeddings)                  # add vectors to the index
# print(index.ntotal)

select = st.sidebar.selectbox("Pick a song.", english_lyrics["song"][:100])
selected_index = english_lyrics[english_lyrics["song"]==select].index[0]

st.write(select)
st.text(english_lyrics[english_lyrics["song"]==select].iloc[0]["lyrics"])

k = 3                          # we want to see 4 nearest neighbors
D, I = index.search(english_embeddings[selected_index:selected_index+1], k) 

for s_no,i in enumerate(I[0]):
    st.text("Similar Song Discovered : " + str(s_no+1))
    st.text(hindi_lyrics.iloc[i]['Song'])