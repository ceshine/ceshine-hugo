---
slug: mpire-postgres-queries
date: 2022-01-07T00:00:00.000Z
title: "Use MPIRE to Parallelize PostgreSQL Queries"
description: "A case study for the high-level Python multiprocessing library"
tags:
  - python
  - tools
  - data_eng
keywords:
  - python
  - tools
  - data engineering
url: /post/mpire-postgres-queries/
---

{{< figure src="/post/sqlite-great-expectations/featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/canal-boats-buildings-reflection-5488271/)" >}}

## Introduction

[Parallel programming is hard](https://blog.mi.hdm-stuttgart.de/index.php/2016/10/24/why-is-parallel-programming-so-hard-to-express/), and you probably should not use any low-level API to do it in most cases (I'd argue that Python's [built-in multiprocessing package](https://docs.python.org/3/library/multiprocessing.html) is low-level). I've been using [Joblib's Parallel class](https://joblib.readthedocs.io/en/latest/parallel.html) for tasks that are [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) and it works wonderfully.

However, sometimes the task at hand is not simple enough for the Parallel class (e.g., you need to share something from the main process that is not pickle-able, or you want to maintain states in each child process). I've recently found this library — [MPIRE (MultiProcessing Is Really Easy)](https://github.com/Slimmer-AI/mpire) — that significantly mitigates this problem of not having enough flexibility, while still having a high-level and user-friendly API.

In the next section, I'll share a case study for MPIRE that would be relevant to data scientists or data engineers who work with databases.

## Case Study

### Problem Description

Imagine that we have a large table in the database, and we want to make thousands of SELECT queries against the table. The WHERE clauses of the queries can look very different. Here are some examples:

```sql
SELECT * FROM table_a WHERE create_time > TIMESTAMP '2021-01-01' and create_time < TIMESTAMP '2021-01-02';
SELECT * FROM table_a WHERE modified_time > TIMESTAMP '2021-01-01' and value_a < 100;
```

(Note that in some simpler cases where the WHERE clauses share the same structure, we can combine the queries into one big query using a temporary table (details in this [Stack Overflow answer](https://stackoverflow.com/questions/65412161/execute-a-query-for-multiple-sets-of-parameters-with-psycopg2)). This approach would maximally utilize the database connection but would require some post-processing of the results. We'll stick with the multi-query approach in the rest of the post as it is more expressive and flexible, despite some overhead on the database connection side.)

The program would waste a lot of time waiting for the database to return the query results if you run these queries sequentially in a single thread, especially when the machine making the queries is not in the same network as the database (e.g., when you're on your laptop running analysis on data in a cloud database). We'll have a much higher throughput if we distribute and run the queries in multiple processes or threads. (In this case, multi-threading would suffice because the operations are I/O-bound, not CPU-bound.)

### Attempt 1: Using Joblib Parallel

We can use this generic query execution function (adapted from [this article](https://medium.com/geoblinktech/parallelizing-queries-in-postgresql-with-python-572995ae340)) and pass it to Joblib:

```python
import pyscopg2
from joblib import Parallel, delayed

# Change this to fit your database:
connect_text = "dbname='%s' user='%s' host=%s port=%s password='%s'" % (dbname, user, host, port, password)

def run_query(connect_text, query, args):
    conn = psycopg2.connect(connect_text)
    with conn.cursor() as cur:
        cur.execute(query, args)
        results = list(cur.fetchall())
    # Uncomment this if you're making changes to the database
    # conn.commit()
    conn.close()
    return results

# This is where the queries and search criteria would go:
payloads = [("query here", ("val1", "val2")), ("query here", ("val3", "val4"))]

results = Parallel(n_jobs=4)(delayed(run_query)(connect_text, query, args) for query, args in payloads)
```

The problem of this solution is that it creates a new connection to the database in every function call (that runs a query) and destroy it afterwards. This creates significant overhead. More ideal way to handle this situation is to create one connection for each worker process/thread, and reuse that connection in every function call sent to the worker. This is not easily achievable in Joblib, as it does not allow you to create a persistent state in a worker.

### Attempt 2: Using MPIRE

[MPIRE's “worker state”](https://slimmer-ai.github.io/mpire/usage/workerpool/worker_state.html) let you initialize worker states with a `worker_init` function before any real work and clean up the states with a `worker_exit` function after all the work are finished.

Let's define the two helper functions:

```python
def init_db_conn(worker_state):
    # `get_db_conn` run the psycopg2.connect and return the resulting connection object
    worker_state['conn'] = get_db_conn()

def close_db_conn(worker_state):
    worker_state['conn'].close()
```

We can now have a much simpler worker function:

```python
def run_query(worker_state, query, args):
    with worker_state['conn'].cursor() as cur:
        cur.execute(query, args)
        results = list(cur.fetchall())
    # Uncomment this if you're making changes to the database
    # worker_state['conn'].commit()
    return results
```

And finally, distribute and run the workloads in multiple threads:

```python
with WorkerPool(
    n_jobs=n_jobs, start_method="threading", use_worker_state=True
) as pool:
    results = pool.map(
        run_query,
        payloads,
        progress_bar=True,
        worker_init=init_db_conn,
        worker_exit=close_db_conn
    )
```

That's it! The API is almost as simple as Joblib Parallel, but also much powerful and versatile. There's a lot more useful features in MPIRE, which you can find out in [their well-written documentation](https://slimmer-ai.github.io/mpire/index.html).

#### Alternative Solution

As a side note, there's a workaround that would allow us to avoid the connection overhead when using Joblib Parallel. We can use `ThreadedConnectionPool` or `PersistentConnectionPool` from the `psycopg2` library to maintain a connection pool that can be shared among worker threads ([more details here](https://pynative.com/psycopg2-python-postgresql-connection-pooling/#h-persistentconnectionpool)). We pass the connection pool object to the worker function and the worker function will obtain a connection from the pool. The connections in the pool are persistent between function calls.

However, this solution depends on the unique API implemented in `psycopg2` and might not be applicable to other libraries and types of databases. It also does not support multi-processing, which might severely impact the performance when we also do some compute-intensive work in the worker function (Python's multi-threading is [hampered by GIL](https://realpython.com/python-gil/)).
