---
slug: text-annotation-1
date: 2019-12-16T00:00:00.000Z
title: "Create a Customized Text Annotation Tool in Two Days - Part 1"
description: "Building a Back-end API Server with Multi-user Support"
tags:
  - nlp
  - dataset
  - fastapi
keywords:
  - nlp
  - dataset
  - annotation
  - fastapi
url: /post/text-annotation-1/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/quotes-excerpts-writing-airbrushed-2608205/)" >}}

# Introduction

In my previous post — [Fine-tuning BERT for Similarity Search](https://blog.ceshine.net/post/finetune-sentence-bert/) — I mentioned that I annotated 2,000 pair of sentence pairs, but did not describe how I did it and what tool I used. Now in this two-part series, we'll see how I created a customized text annotation tool that greatly speeds up the annotation process.

The entire stack is developed in two days. You can probably do it a lot faster if you are familiar with the technology (the actual time I spent on it is about 6 hours top).

## Why Build Your Own Annotation Tool

You might ask, why build your own tool? Why not just use Excel or open-source/proprietary tools? There are several reasons:

1. The Excel UI is not necessarily the best fit for test reading and annotation.
1. You have to do all the data management inside Excel, including shuffling, sampling, or splitting the dataset. All of them are tedious manual labor and prone to errors.
1. Open-source tools can be overly complicated for your use case, and also requires a lot of work to customize.
1. You might not want to pay for the proprietary tool if the scale of your annotation work is relatively small.
1. Learning — you can see this as an opportunity to practice and improves your web development skills. (More details in the next section.)

The customized annotation tool I built does these things:

1. Manage user sessions — uses signed cookies to identify users, provide different sets of pairs for annotation, and keep track of annotated pairs.
1. Editable annotations — users can submit the results, make some changes afterward, and submit them again. No duplicated entries will be generated.
1. Each submission is stored in a CSV file, with a timestamp field — you can easily pick out annotations from a specific time range that could be problematic.
1. Automatically pick a score based on the model prediction — the annotator only needs to adjust the model prediction. Also, it gives you an idea of how the model performs on unseen entries.

{{< figure src="screenshot.png" caption="Our Annotation Interface. Need some work aesthetically, but useful enough IMO." >}}

## Inspiration and Learning Resources

I don't remember exactly where I got the idea of a React front-end working with a Python back-end from. I think it was from a tweet by Ines Montani, but here's a more recent reminder by Joel Grus:

{{< single_tweet 1143951419151540225 >}}

Having React in your arsenal is very empowering. You don't need to attend a boot camp or spend a couple of months. Just learn the basics and start building stuffs. Granted, the results can be very ugly (in terms of both presentation and code), but as long as it works reasonably well, it's still better than nothing. (You'll still probably need professional help if you want to build any customer-facing applications.)

The learning resources in the following Twitter thread by Ines Montani are very helpful:

{{< single_tweet 1144173215293591555 >}}

# The API Server

We're going to use [FastAPI](https://github.com/tiangolo/fastapi) to build our back-end API server. It's created by Sebastián Ramírez (tiangolo) and is built on top of [Starlette](https://www.starlette.io/). The other popular alternative is to use [Flask](https://www.palletsprojects.com/p/flask/). I had experience using both, and IMO FastAPI is better if you just want to build an API server (without any no front-end logic).

The source code can be found at [veritable-tech/text-annotation-api-server/server.py](https://github.com/veritable-tech/text-annotation-api-server/blob/blog-post/server.py). The codebase could use some refactoring, but this state represents how I quickly put together all the necessary parts in the first place.

## Creating Session

(I'm going to skip the non-essential lines here to save space. Please refer to the source code for the full implementation).

First, we need to create the app object and add a session middleware:

```python
from starlette.middleware.sessions import SessionMiddleware
from fastapi import FastAPI
APP = FastAPI()

APP.add_middleware(
    SessionMiddleware,
    secret_key="PUT_YOUR_SECRET_KEY_HERE"
)
```

The `secret_key` is used to sign the cookies, which makes sure their contents are not tampered with.

The next step is to create a GET endpoint for retrieving a batch/set of sentence pairs to annotate. At the start of the function, we check if we've seen this user. If not, we create a new user ID for them and save the ID to the associated session:

```python
@APP.get("/batch/", response_model=BatchForAnnotation)
def get_batch(request: Request):
    if not request.session or request.session["uid"] not in GLOBAL_CACHE:
        print("Creating new indices...")
        request.session["uid"] = str(uuid.uuid4())
        # Shuffle the dataset (via shuffling the index)
        indices = np.arange(len(DATASET))
        np.random.shuffle(indices)
        # Initialize the user states
        GLOBAL_CACHE[request.session["uid"]] = {
            "indices": indices,
            "submitted": {}
        }
```

Since I only run the annotation tool locally and on a small scale, I keep all the application states in a global dictionary object `GLOBAL_CACHE`. For a more robust solution, you can consider using something like [Redis](https://redis.io/) to store the states.

The following is an example of how we use the user ID in session data and the application states to reject the POST request from users who have not fetched any batch yet:

```python
@APP.post("/batch/", response_model=SubmitResult)
def submit_batch(batch: BatchAnnotated, request: Request):
    if (
        not request.session.get("uid") or
        not request.session.get("uid") in GLOBAL_CACHE
    ):
        return SubmitResult(
            success=False,
            overwrite=False,
            message="You haven't fetched any batches yet."
        )
```

## Loading the Dataset

This is how I prepare the dataset: I have 8,000+ very short paragraphs (in Traditional Chinese), and an existing baseline model (Multilingual Universal Sentence Encoder or a fine-tuned model). I use the model to collect the 20 ~ 30 most similar paragraphs and another 20 ~ 30 random paragraphs for each paragraph. I save the result into a CSV file with four fields — "text_1", "text_2", "similarity", "similarity_raw". (The "similarity_raw" field is the score before the optional linear transformation. It is there just for reference.)

The CSV file is loaded before launching the application:

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data-path', type=str, default="data/dataset.csv")
    args = parser.parse_args()
    DATASET = pd.read_csv(args.data_path)
    print(f"Listening to port {PORT}")
    uvicorn.run(APP, host='0.0.0.0', port=PORT)
```

## Preparing a Batch

This is how each batch is prepared:

```python
def find_page_to_annotate(cache_entry: Dict):
    for page in range(int(math.ceil(len(DATASET) / PAIRS_PER_PAGE))):
        if page not in cache_entry["submitted"]:
            return page
    return None

class BatchForAnnotation(BaseModel):
    page: int
    pairs: List[Tuple[int, str, str, float]] = []

@APP.get("/batch/", response_model=BatchForAnnotation)
def get_batch(request: Request):
    # Skipped ...
    cache_entry = GLOBAL_CACHE[request.session["uid"]]
    page = find_page_to_annotate(cache_entry)
    if page is None:
        return BatchForAnnotation(
            page=-1,
            pairs=[]
        )
    batch = DATASET.loc[
        cache_entry["indices"][
            page*PAIRS_PER_PAGE:(page+1)*PAIRS_PER_PAGE
        ]
    ]
    pairs = list(batch[
        ["text_1", "text_2", "similarity"]
    ].itertuples(index=True, name=None))
    print(f"Page: {page} Items: {len(pairs)}")
    return BatchForAnnotation(
        page=page,
        pairs=pairs
    )
```

The `GLOBAL_CACHE[uid]["submitted"]` store `page -> output_path` key/value pairs. The `output_path` points to a CSV file where a user-submitted batch of annotations has been saved to.

The `find_page_to_annotate` function finds the first page that hasn't been annotated yet.

## Accepting a Batch Submission

When a user finished annotating the received batch, they submit the results via a POST request. Only the ID of the pair and the annotated similarity score is submitted (as we already have the corresponding text in the memory):

```python
class SubmitResult(BaseModel):
    success: bool
    overwrite: bool
    message: str

class BatchAnnotated(BaseModel):
    page: int
    pairs: List[Tuple[int, float]] = []

@APP.post("/batch/", response_model=SubmitResult)
def submit_batch(batch: BatchAnnotated, request: Request):
    # Skipped...
    page = batch.page
    batch_orig = DATASET.loc[
        GLOBAL_CACHE[request.session["uid"]]["indices"][
            page*PAIRS_PER_PAGE:(page+1)*PAIRS_PER_PAGE
        ]
    ].copy()
    indices, similarities = list(zip(*batch.pairs))
    # Skipped the data validation block...
    batch_orig.loc[indices, "similarity"] = similarities
    batch_orig["timestamp"] = int(datetime.now().timestamp())
    overwrite = False
    if page in GLOBAL_CACHE[request.session["uid"]]["submitted"]:
        # Overwrite a submitted batch
        output_path = GLOBAL_CACHE[request.session["uid"]]["submitted"][page]
        overwrite = True
    else:
        # This is a new batch
        output_path = OUTPUT_DIR / \
            f"{datetime.now().strftime('%Y%m%d_%H%M')}_{page}.csv"
        GLOBAL_CACHE[request.session["uid"]]["submitted"][page] = output_path
    batch_orig.to_csv(output_path, index=False)
    return SubmitResult(
        success=True,
        overwrite=overwrite,
        message=""
    )
```

The code should be quite straight-forward to read. We create a copy of a slice of the Pandas data frame, do some data validation to make sure the pair IDs and the page are matched, and update the slice with the submitted labels. If the submitted page already has been submitted before, we overwrite the previous output; if not, we create a new output file.

## Potential Improvements

We now already have a functioning back-end server with just about 150 lines of code. There are many potential improvements, including:

- Specify which page to fetch in the GET request. (Currently, we can only fetch the pages that have not received any submissions.)
- Store the user ID in the output file.
- Use a persistent store for the application states, so an application restart will not erase previous records.
- Customizable page size at the start of the session.

Some of these are relatively easy to implement. Readers are encouraged to implement them as exercises.

# To Be Continued

We've covered the back-end API server in this post. In the next post, we'll describe how to write a fairly basic React front-end to interact with both the user and the back-end server.

[Read the Part 2 here](https://blog.ceshine.net/post/text-annotation-2/).
