---
slug: python-cron-telegram-wrapper
date: 2020-04-10T00:00:00.000Z
title: "Monitor Python Script Cron Jobs using Telegram"
description: "Get a Message when a Job starts, finishes, or crashes."
tags:
  - python
  - tips
keywords:
  - python
  - telegram
  - cron
url: /post/python-cron-telegram-wrapper/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/city-window-architecture-urban-4991094/)" >}}

# Motivation

[Apache Airflow](http://airflow.apache.org/) is great for managing scheduled workflows, but in a lot of cases, it is an overkill and brings unnecessary complexity to the overall solution. [Cron](https://www.wikiwand.com/en/Cron) jobs are much easier to set up, have built-in support in most systems, and have a very flat learning curve. However, the lack of monitoring features and the consequential silent failures can be the bane of system admins' lives.

We want a simple solution that can help admins monitor the health of cron jobs in simple scenarios that do not warrant Airflow. The simple scenarios have the following characteristics:

1. Only a handful of jobs to monitor. A good rule of thumb: fewer than 12 cumulative job runs per day.
2. No dependencies between jobs.
3. Compute resources are limited. We don't want to host another back-end server just to monitor the cron jobs.
4. The job is straight-forward or well-tested, so we rarely have to read the log file.
5. The job is not mission-critical, which typically requires you to respond to 100% of the execution errors and respond within an hour. Examples include periodic maintenance or backup scripts.

We also only focus on running python scripts as cron jobs. You can always wrap shell scripts inside Python, so this shouldn't be a big problem.

## Honorable Mention: Cronhub

[Cronhub](https://cronhub.io/) is a SaaS that lets you get instant alerts when any of your background jobs fail silently or run longer than expected. You only need to append a Ping API call at the end of your cron command. It's simple yet powerful.

There are several advantages of the Telegram solution proposed below over Cronhub:

1. **Python-specific messages**. Cronhub is designed for generic jobs, and there is no way to pass additional information about the job execution(e.g., the error messages or the return values.).
2. Not depending on external services (other than the Telegram API).
3. The monitoring scheme is stored in the codebase.

# Solution: A Python Decorator

My solution is to create a decorator called `telegram_wrapper` that is heavily based on the `telegram_sender` from [huggingface/knockknock](https://github.com/huggingface/knockknock/blob/master/knockknock/telegram_sender.py). The knockknock implementation is designed for training machine learning models. I tweaked the message templates and added a few options to make it more suitable for monitoring background tasks.

## Telegram Bot

You need to have a [Telegram client](https://telegram.org/apps) and create a [Telegram bot](https://core.telegram.org/bots#6-botfather). Send the first message to the bot, and then use the token of your bot to find your chat_id by visiting [https://api.telegram.org/bot<YourBOTToken>/getUpdates](https://api.telegram.org/bot<YourBOTToken>/getUpdates).

## Usage

First, you need to install `cronhelpers`:

```bash
pip install https://github.com/ceshine/cronhelpers/archive/master.zip
```

And wrapper the function that contains the cron job with `telegram_wrapper`:

```python
from cronhelpers import telegram_wraooer

# This one only sends a message when the cron job failed
@telegram_wrapper(
  "your_token", "your_chat_id", name="jobName",
  send_at_start=False, send_on_success=False)
def simple_func(arg):
    return arg

if __name__ == main():
    simple_func()
```

Then set up the crontab as usual:

```text
10 0,12 * * *  /path/to/python some_script.py
```

Usually, I would containerize the python environment, and the crontab entry would look like this:

```text
10 0,12 * * *  docker run --rm somecontainer >> /home/ceshine/somejob.log 2>&1
```

{{< figure src="example.png" caption="Example message on successful execution." >}}

## Notes

1. Set `send_at_start=True` to get a message when a job starts. This is usually not necessary but could be useful if it's a long-running job and you'd like to see the job starts at say 8 pm sharp.
2. Set `send_on_success=False` to only get a message when a job crashes. Beware that if the job is killed or the machine is shut down you still get no message at all. So use it only when your machine is stable and not crowded.
3. The `telegram_wrapper` should be able to catch any exceptions raised by the wrapped function, but it is impossible to handle hard crashes (job killed or machine shut down) inside Python. The only way to detect a hard crash is to read the Telegram chat log to see if the expected message shows up. Hopefully, this happens only occasionally. And please **make sure your job is tested on the target machine** before writing it into the crontab.
