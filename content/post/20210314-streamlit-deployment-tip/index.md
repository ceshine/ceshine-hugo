---
slug: streamlit-deployment
date: 2021-03-14T00:00:00.000Z
title: "Mistake I Made that Crippled My Streamlit App"
description: "Not properly caching slows down the app and increases memory consumption"
tags:
  - python
  - streamlit
  - pytorch
  - tip
keywords:
  - python
  - streamlit
  - pytorch
  - tip
url: /post/streamlit-deployment/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/people-religion-art-statue-street-3299174/)" >}}

[Streamlit](https://streamlit.io/) an increasingly popular tool that allows Python developers to turn data scripts into interactive web applications in a few lines of code. I recently developed and deployed a [semantic search app](https://news-search.veritable.pw/) for news articles in Chinese, and I made a mistake not caching the model loading code. The performance was abysmal, and the memory footprint was huge for a TinyBERT-4L model (had to allocate 1GB of memory for the app).

Thankfully, Randy Zwitch(@randyzwitch) from Streamlit pointed out the problem to me on Twitter:

{{< single_tweet 1368917428911083523 >}}

By correctly caching the FAISS index and creating a separate API server to serve the PyTorch model (to support other use cases), I managed to increase the speed dramatically  and memory consumption of the app (now only need to allocate a most 600MB of memory in total):

{{< figure src="docker_stats.jpg" caption="Memory consumption of my Docker containers" >}}

## Elaboration

### The Mistake

Here’s a more extended version of the story. I’m used to the way Flask and FastAPI handle things — the models are loaded as global variables, the caching is only required for serving repeating inputs that require heavy computation or significant I/O latency.

As an example, the following is an excerpt of [this FastAPI script](https://github.com/ceshine/oggdo/blob/1ac3d4ff0eb2332740297eb67db55d7ed6cd24e5/api-server/main.py) that returns sentence embedding vectors of the input sentences:

```python
APP = FastAPI()

if os.environ.get("MODEL", None):
    MODEL = SentenceEncoder(os.environ["MODEL"], device="cpu")

@APP.post("/", response_model=EmbeddingsResult)
def get_embeddings(text_input: TextInput):
    assert MODEL is not None, "MODEL is not loaded."
    text = text_input.text.replace("\n", " ")
    if text_input.t2s:
        text = T2S.convert(text)
    vector = MODEL.encode(
        [text],
        batch_size=1,
        show_progress_bar=False
    )[0]
    return EmbeddingsResult(vectors=vector.tolist())
```

When developing the Streamlit app, I did something similar to this:

```python
encoder = SentenceEncoder(
    "streamlit_model/", device="cpu"
).eval()

def main():
    st.title('Veritable News Semantic Search Engine')
    # ...
    if len(query) > 10 and len(date_range) == 2:
        embs = encoder.encode(
            [query],
            batch_size=1,
            show_progress_bar=True
        )
        #...
```

Because Streamlit runs the entire script for every interaction (and some other events, it seems), the model gets repeatedly loaded into the memory (one can verify that from the log), bringing devastating results. When running on my low-end Hetzner VPS instance, the app sometimes simply failed to load (stuck at the initial “connecting” animation).

## The Solution

The correct way to load large objects from the disk is to use the [@st.cache](https://docs.streamlit.io/en/stable/api.html#streamlit.cache) function decorator. It's a simple memory cache for Streamlit's unique architecture. Streamlit stores the results from that function in the first run and provides the reference to the results in the subsequent runs.

The following is taken from this script, with some modifications (I put the sentence encoder back in):

```python
@st.cache(allow_output_mutation=True)
def load_data():
    conn = sqlite3.connect("data/news.sqlite")    
    full_ids = joblib.load("data/ids.jbl")
    index = faiss.read_index("data/index.faiss")
    default_date_range = [datetime.date(2018, 11, 28), datetime.date.today()]
    encoder = SentenceEncoder(
        "streamlit_model/", device="cpu"
    ).eval()    
    return conn, full_ids, index, default_date_range, encoder
```

One thing to emphasize again is that Streamlit provides **references** to cached function results in every run.  No copying is involved, so the memory requirement will not be doubled when caching is enabled. In fact, repeatedly reloading the model turns out to have larger memory footprints in my observations. The reason behind it is possibly the garbage collecting mechanism of Python. When you rapidly put new things into memory, the garbage collector might not have time to release the space used by the previous objects fast enough.
