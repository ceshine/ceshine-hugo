---
slug: gcp-shutdown-script
date: 2019-10-25T00:00:00.000Z
title: "Pro Tip: Use Shutdown Script Detect Preemption on GCP"
description: "Get a Notification when your Google Cloud Compute instance is preempted"
tags:
  - tip
  - gcp
keywords:
  - tutorial
  - gcp
url: /post/gcp-shutdown-script/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/panda-red-panda-bear-cat-4546244/)" >}}

# Motivation

I was recently given a $300 credit to [Google Cloud Platform](https://cloud.google.com/) from Google to participate in a Kaggle competition. This gives me free access to the powerful GPUs (T4, P100, even V100) to train models and thus opens the window to many new possibilities. However, the problem is that $300 can be used up rather quickly. For example, [one Tesla P100 GPU cost \$1.46 per hour](https://cloud.google.com/compute/gpus-pricing), so \$300 can only give me 200 hours or 8.5 days. Don't forget there are still other costs from CPU, memory, and disk storage.

The good news is Google provides a much cheaper option: [preemptible instances](https://cloud.google.com/preemptible-vms/). Preemptible instances can run for up to 24 hours, and can be preempted (shut down) at any time. The preemptible GPUs are about [70% cheaper](https://cloud.google.com/compute/gpus-pricing) than regular ones. That translates to more than 3 times more hours to build and train your models.

Resumable training is essential when using pre-emptible instances. Regularly saves checkpoints during training, and you can resume training from the latest checkpoint when the instance is preempted or shut down after 24 hours of running. Please check the documentation of your deep learning framework to see how to do resumable training.

The last piece of puzzle is **getting a notification when your preemptible instance is being preempted**. Otherwise, you will have to stare at the terminal or the browser tab (if you're using Jupyter notebook) during training to know when to restart the instance and resume training. This short post will introduce a way to do this via shutdown scripts.

# Solution

[Google Cloud Compute](https://cloud.google.com/compute/) allows users to run [startup script](https://cloud.google.com/compute/docs/startupscript) and [shutdown script](https://cloud.google.com/compute/docs/shutdownscript) via its [metadata server](https://cloud.google.com/compute/docs/storing-retrieving-metadata). Preemptible instances will have 30 seconds to run the shutdown script after the shutdown process begins, which is more than enough for our use case — sending a notification.

```python
#!/home/ceshine/miniconda3/envs/pytorch/bin/python
import socket
import telegram

BOT_TOKEN = "bot token here"
CHAT_ID = "chat id here"

bot = telegram.Bot(token=BOT_TOKEN)
host_name = socket.gethostname()
content = 'Machine name: %s is shutting down!' % host_name
bot.send_message(chat_id=CHAT_ID, text=content)
```

Above is a very simple shutdown script sending out a Telegram notification with `python-telegram-bot`. **Modify the first line to the path to the Python binary you're using, and make sure `python-telegram-bot` is installed in its environment.** Please follow [this document](https://github.com/python-telegram-bot/python-telegram-bot/wiki/Extensions-%E2%80%93-Your-first-Bot) to set up the Telegram bot if you don't have one yet. Fill in the bot token and chat id in the above script.

Now you have the script. The next step is to assign them to the preemptible instances. You'll need [the gcloud command-line tool](https://cloud.google.com/functions/docs/quickstart) for this (you can use the GUI interface, but I find using the command line most convenient).

If you haven't created the instance yet, simply add a `--metadata-from-file` parameter to the instance creation command like this:

```bash
gcloud compute instances create example-instance \
    --metadata-from-file shutdown-script=examples/scripts/telegram_notification.py
```

If you already created the instance, run the following command to set the shutdown script for that instance:

```bash
gcloud compute instances add-metadata instance-name  \
    --metadata-from-file shutdown-script=examples/scripts/telegram_notification.py
```

And that's it! You'll receive a Telegram notification when the instance is shutting down.

Note: you can also upload the script to a Google Cloud Storage bucket and attach to an instance like this:

```bash
gcloud compute instances create example-instance --scopes storage-ro \
    --metadata shutdown-script-url=gs://bucket/startupscript.sh
```

## Shutdown script invocation

According to the GCP documentation, a shutdown script runs as part of the following actions:

- When an instance shuts down due to an `instances.delete` request or an `instances.stop` request to the API.
- When Compute Engine stops a preemptible instance as part of the preemption process.
- When an instance **_shuts down through a request to the guest operating system_**, such as sudo shutdown or sudo reboot.
- When you shut down an instance manually through the GCP Console or the gcloud compute tool.

(The shutdown script won't run if the instance is reset using `instances().reset`.)

So the shutdown script will be invoked even when you manually shut down the instance, i.e., you'll get a Telegram notification of your instance being shut down. That shouldn't be a big problem because you already know you are shutting down the instance.

# Conclusion

This post describes a way to configure shutdown scripts so you'll be notified when one of your instances is being preempted. This simple time-saving trick can help you minimize the time wasted between the preemption and the resuming of training.

Granted, this trick is useless when you are asleep or have no access to a computer. A much more sophisticated solution is required for these scenarios. The shutdown script will have to robustly restart the instance and resume training automatically. Personally I'm happy to give up supporting those scenarios entirely. But it is up you to decided if you want to spend time implementing that.
