---
slug: tf21-quest
date: 2020-02-13T00:00:00.000Z
title: "Tensorflow 2.1 with TPU in Practice"
description: "Case Study: Google QUEST Q&A Labeling Competition"
tags:
  - nlp
  - tensorflow
  - kaggle
keywords:
  - nlp
  - tensorflow
  - tpu
  - kaggle
url: /post/tf21-quest/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/the-side-of-the-road-snow-mountains-4259510/)" >}}

# Executive Summary

- **Tensorflow has become much easier to use**: As an experience PyTorch developer who only knows a bit of Tensorflow 1.x, I was able to pick up Tensorflow 2.x using my spare time in 60 days and do competitive machine learning.
- **TPU has never been more accessible**: The new interface to TPU in Tensorflow 2.1 works right out of the box in most cases, and greatly reduces the development time required to make a model TPU-compatible. Using TPU drastically increases the speed of experiment iterations.
- **We present a case study of solving a Q&A labeling problem by fine-tuning RoBERTa-base model from _huggingface/transformer_ library**:
  - [Codebase](https://github.com/ceshine/kaggle-quest)
  - [Colab TPU training notebook](https://gist.github.com/ceshine/752c77742973a013320a9f20384528a1)
  - [Kaggle Inference Kernel](https://www.kaggle.com/ceshine/quest-roberta-inference/data?scriptVersionId=28553401)
  - [High-level library TF-HelperBot](https://github.com/ceshine/tf-helper-bot/) to provide more flexibility than the Keras interface.
- (Tensorflow 2.1 and TPU are also very good fit for CV applications. A case study of solving a image classification problem will be published in about a month.)

# Acknowledgement

I was granted free access to Cloud TPUs for 60 days via Tensorflow Research Cloud. It was for the [TensorFlow 2.0 Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering) competition. I chose to do this simpler [Google QUEST Q&A Labeling](https://www.kaggle.com/c/google-quest-challenge) competition first, but unfortunately couldn't find enough time to go back and do the original one (sorry!).

I was also granted \$300 credits for the [TensorFlow 2.0 Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering) competition, and had used those to develope a [PyTorch baseline](https://github.com/ceshine/kaggle-tf2qa). They also covered the costs of Cloud Compute VM and Cloud Storage used to train models on TPU.

# Introduction

Google was handing out free TPU access to competitors in the [TensorFlow 2.0 Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering) competition, as an incentive for them to try out the newly added TPU support in Tensorflow 2.1 (then RC). Because the preemptible GPUs on GCP are barely usable at the time, I decided to give it a shot. It all began with this tweet:

{{< single_tweet 1210092823564832768 >}}

Turns out that the Tensorflow model in [huggingface/transformers](https://github.com/huggingface/transformers) library can work with TPU without modification! I then proceeded to develop models using Tensorflow(TF) 2.1 for a simpler competition [Google QUEST Q&A Labeling](https://www.kaggle.com/c/google-quest-challenge).

I missed the post-processing trick in the QUEST competition because I spent most of my limited time wrestling with TF and TPU. After applying the post-processing trick, my final model would be somewhat competitive at around 65th place (silver medal) on the final leaderboard. **The total training time of my 5-fold models using TPUv2 on Colab was about an hour**. This is a satisfactory result in my opinion, given the time constraint.

## Tensorflow 2.x

The Tensorflow 2.x has become much more approachable, and the customized training loops provide a swath of opportunities to do creative things. I'm more confident now that I'll be able to re-implement top solutions of the competition in TF 2.x without banging my head on the door (at least less frequently).

On the other hand, TF 2.x is still not as intuitive as PyTorch. The documentation and community support still has much to be desired. Many of the search results still point to TF 1.x solutions which are not applicable to TF 2.x.

As an example, I ran into this problem in which the CuDNN failed to initialize:

{{< single_tweet 1227289969896505344 >}}

One of the solution is to limit the GPU memory usage, and here's a confusingly long thread on how to do so:

{{< single_tweet 1227289971368685568 >}}

## TPU Support

Despite of all the drawbacks, the TPU support in TF 2.1 is fantastic, and has become my main reason of using TF 2.1+. It's hard to imagine how the extremely unstable TPU support in Keras has evolved into this good piece of engineering.

Although Tensorflow Research Cloud gave my access to multiple TPUs, I used only one of them as I didn't see the need to do serious hyper-parameter optimization yet. The competition data set is not ideal for TPU, as it is quite small (a few thousands of examples). I have to limit the batch size to achieve the best performance (in terms of the evaluation metric), but it is still a lot faster than training on my single local GTX 1070 GPU (4 ~ 8x speedup). TPUv2 is more than sufficient in this case (comparing to TPUv3).

One potentially interesting comparison would be using two V100 GPUs, which combined are a little more expensive than TPUv2, with a bigger batch size to train the same model.

**The TPU on Google Colab also supports TF 2.1 now**. You are able to train models much faster with it than any of the free GPU Colab provides (currently the best offer is a single Tesla P100). Check this notebook for a concrete example:

{{< single_tweet 1215182582083538944 >}}

(I know that PyTorch has [its own TPU support](https://github.com/pytorch/xla) now, but it is still quite hard to use last time I check, and it is not supported in Google Colab. Maybe I'll take another look in the next few weeks.)

# Case Study and Code Snippets

We'll briefly describe my solution to the QUEST Q&A Labeling competition, and discuss some parts of the code that I think are most helpful for those who come from PyTorch like I did. This section assumes that you already have a basic understanding of Tensorflow 2.x. If you're not sure, please refer to the official tutorial [Effective TensorFlow 2](https://www.tensorflow.org/guide/effective_tf2).

- [Codebase](https://github.com/ceshine/kaggle-quest)
- [Colab TPU training notebook](https://gist.github.com/ceshine/752c77742973a013320a9f20384528a1)
- [Kaggle Inference Kernel](https://www.kaggle.com/ceshine/quest-roberta-inference/data?scriptVersionId=28553401)
- [High-level library TF-HelperBot](https://github.com/ceshine/tf-helper-bot/) to provide more flexibility than the Keras interface.

## Roadmap

1. TF-Helper-Bot: this is a simple high-level wrapper of Tensorflow I wrote to improve code reusability.
2. Input Formulation and TFRecords Preparation.
3. TPU-compatible Data Loading.
4. The Siamese Encoder Network.

## TF-Helper-Bot

[TF-Helper-Bot](https://github.com/ceshine/tf-helper-bot/) is a simple high-level wrapper of Tensorflow, and is basically a port of my other project — [PyTorch-Helper-Bot](https://github.com/ceshine/pytorch-helper-bot)(which is heavily inspired by the fast.ai library). It handles custom training loops, distributed training (TPU), metric evaluation, checkpoints, and some other useful stuffs for you.

### BaseBot

The central component of TF-Helper-Bot is the `BaseBot` class. It would normally be inherited and adapted for each new project. Think of it as a robot butler. Give her/him a model, an optimizer, a loss function, and other optional goodies via `__init__()`, and call `train()`. The robot will have your model trained and ready. You can also call `eval()` to do validation or testing, and call `predict()` to make predictions.

One important ways to improve Tensorflow 2.x code performance is to use [tf.function](https://www.tensorflow.org/guide/function) to mark a function for JIT compilation. This is how TF-Helper-Bot does it ([source code location](https://github.com/ceshine/kaggle-quest/blob/795d94c70f7c97fd2c0a5f383fdf52571f9bb0ed/tf-helper-bot/tf_helper_bot/bot.py#L13)):

(`BaseBot` uses [_dataclass_](https://docs.python.org/3.7/library/dataclasses.html) to manage internal instance states. Therefore, the additional initialization code goes into `__post_init__()` method)

```python
@dataclass
class BaseBot:
    train_dataset: tf.data.Dataset
    # omitted...
    criterion: Callable
    model: tf.keras.Model
    optimizer: tf.keras.optimizers.Optimizer
    name: str = "basebot"
    # omitted...

    def __post_init__(self):
        # omitted...

        @tf.function
        def get_gradient(input_tensors, target):
            with tf.GradientTape() as tape:
                output = self.model(
                    input_tensors, training=True)
                loss_raw = self.criterion(
                    target, self._extract_prediction(output)
                )
                loss_ = (
                    self.optimizer.get_scaled_loss(loss_raw)
                    if self.mixed_precision else loss_raw
                )
            gradients_ = tape.gradient(
                loss_, self.model.trainable_variables)
            if self.mixed_precision:
                gradients_ = self.optimizer.get_unscaled_gradients(gradients_)
            return loss_raw, gradients_

        @tf.function
        def step_optimizer(gradients):
            self.optimizer.apply_gradients(
                zip(
                    gradients,
                    self.model.trainable_variables
                )
            )

        @tf.function
        def predict_batch(input_tensors):
            return self.model(input_tensors, training=False)

        self._get_gradient = get_gradient
        self._step_optimizer = step_optimizer
        self._predict_batch = predict_batch
```

- The reason why it uses seemingly contrived nested function is that decorating the class methods with `tf.function` doesn't work for me.
- Even decorating the bland `predict_batch` can improve the performance significantly. Without this I couldn't get the inference kernel to finish within the time limit in another CV competition.
- The `get_gradient` method supports mixed precision training, which won't be covered in this post.

### BaseDistributedBot

If we want to do distributed training (TPU has 8 cores), we'll nee some specialized interfaces to do that. We accommodate these by subclassing ([source code location](https://github.com/ceshine/kaggle-quest/blob/795d94c70f7c97fd2c0a5f383fdf52571f9bb0ed/tf-helper-bot/tf_helper_bot/bot.py#L249)):

```python
@dataclass
class BaseDistributedBot(BaseBot):
    strategy: tf.distribute.Strategy = None

    def __post_init__(self):
        assert self.strategy is not None
        assert self.gradient_accumulation_steps == 1, (
            "Distribution mode doesn't suppoprt gradient accumulation"
        )
        super().__post_init__()
        @tf.function
        def train_one_step(input_tensor_list, target):
            loss, gradients = self._get_gradient(
                input_tensor_list[0], target)
            self._step_optimizer(gradients)
            return loss

        self._train_one_step = train_one_step

    def train_one_step(self, input_tensors, target):
        loss = self.strategy.experimental_run_v2(
            self._train_one_step,
            args=(input_tensors, target)
        )
        return self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, loss, axis=None
        )
```

The TPU requires that the entire training loop to be compiled into graphs (i.e. the function you pass `experimental_run_v2` must be compiled), and I couldn't find a way to do gradient accumulation in such situation. As a result, gradient accumulation is removed and the `train_one_step()` method has been simplified. One additional note: when I combined `_get_gradient` and `step_optimizer`, they both would be automatically compiled, so they don't really need their own `tf.function` decorator.

In the case of TPU, you'll need to initialize the TPU using `tf_hepler_bot.utils.prepare_tpu`([source code location](https://github.com/ceshine/kaggle-quest/blob/795d94c70f7c97fd2c0a5f383fdf52571f9bb0ed/tf-helper-bot/tf_helper_bot/bot.py#L249)):

```python
def prepare_tpu():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
        tpu = None
    strategy = tf.distribute.get_strategy()
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    return strategy, tpu
```

## TFRecords Preparation

The QUEST training dataset contains just over 6,000 question and answer pairs, and there are only around 3,500 unique questions. Each question has a title and a body field.

I split the pair into two sequences — question and answer. This is how I formulate the question sequence:

```text
<s> question </s><s> this is the title of a question </s><s> this is the body of a question </s>
```

And the answer sequence:

```text
<s> answer </s><s> this is the body of an answer </s>
```

(I padded the sequence with some whitespace to make it more readable.)

The `<s>` and `</s>` is what [RoBERTa](https://arxiv.org/abs/1907.11692) uses to mark a sentence. The `<s> question </s>` and `<s> answer </s>` header is to help the encoder distinguish between the two types of sequences.

The input data pipeline used by TPU must not contain any python code, and the TPU does not support reading from your local filesystem (the TPU is connected to your VM via network). The simplest way to create a input pipeline for TPU is to save your data into TFRecord files and store it in a [Cloud Storage](https://cloud.google.com/storage/) bucket. TPU will read directly from your bucket and run the compiled pipeline.

(Theoretically if your dataset fit into memory, you can create it locally in memory and send it over to TPU. But I haven't found any good examples of this approach yet. However, always dumping your dataset into TFRecord files for more consistency is a good practice in my opinion.)

This is how I convert the tokenized sequences and labels into TFRecord files[source code location](https://github.com/ceshine/kaggle-quest/blob/795d94c70f7c97fd2c0a5f383fdf52571f9bb0ed/quest/prepare_tfrecords.py#L107):

```python
def to_example(input_dict, labels):
    feature = {
        "input_ids_question": tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=input_dict["input_ids_question"])
        ),
        "input_mask_question": tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=input_dict["input_mask_question"])
        ),
        "input_ids_answer": tf.train.Feature(
            int64_list=tf.train.Int64List(value=input_dict["input_ids_answer"])
        ),
        "input_mask_answer": tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=input_dict["input_mask_answer"])
        ),
        "labels": tf.train.Feature(
            float_list=tf.train.FloatList(value=labels)
        )
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _write_tfrecords(inputs, labels, output_filepath):
    with tf.io.TFRecordWriter(str(output_filepath)) as writer:
        for input_dict, labels_single in zip(inputs, labels):
            example = to_example(input_dict, labels_single)
            writer.write(example.SerializeToString())
    print("Wrote file {} containing {} records".format(
        output_filepath, len(inputs)))
```

## Data Loading

Tensorflow as a [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) module specifically for creating input pipelines. It is especially useful when writing pipeline that will compiled into graphs (which is required by TPU). The following is how I use it to load the data from the TFRecord files (with some details omitted)([source code location](https://github.com/ceshine/kaggle-quest/blob/795d94c70f7c97fd2c0a5f383fdf52571f9bb0ed/quest/dataset.py#L7)):

```python
AUTOTUNE = tf.data.experimental.AUTOTUNE

def tfrecord_dataset(filename, batch_size, strategy, is_train: bool = True):
    opt = tf.data.Options()
    opt.experimental_deterministic = False

    # omitted...

    features_description = {
        "input_ids_question": tf.io.FixedLenFeature([max_q_len], tf.int64),
        "input_mask_question": tf.io.FixedLenFeature([max_q_len], tf.int64),
        "input_ids_answer": tf.io.FixedLenFeature([max_a_len], tf.int64),
        "input_mask_answer": tf.io.FixedLenFeature([max_a_len], tf.int64),
        "labels": tf.io.FixedLenFeature([30], tf.float32),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        example = tf.io.parse_single_example(
            example_proto, features_description)
        return (
            {
                'input_ids_question': tf.cast(example['input_ids_question'], tf.int32),
                'attention_mask_question': tf.cast(example['input_mask_question'], tf.int32),
                'input_ids_answer': tf.cast(example['input_ids_answer'], tf.int32),
                'attention_mask_answer': tf.cast(example['input_mask_answer'], tf.int32),
            },
            example["labels"]
        )

    raw_dataset = tf.data.TFRecordDataset(
        filename, num_parallel_reads=4
    ).with_options(opt)
    dataset = raw_dataset.map(
        _parse_function, num_parallel_calls=AUTOTUNE
    ).cache()
    if is_train:
        dataset = dataset.shuffle(
            2048, reshuffle_each_iteration=True
        ).repeat()
    else:
        # omitted...
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    #omitted...
```

- We've padded the token sequences to a fixed length when creating TFRecords, so `tf.io.FixedLenFeature` is used to parse the feature. If we want to use sequence of variable lengths, we should use `tf.io.FixedLenSequenceFeature`. (More details later.)
- TPU only supports 32-bit integers, and the TFRecord only support 64-bit integers, so we need to do a conversion.
- `cache()` method is called to save the parsed data into memory, and should be able to avoid parsing the same data over and over again. (You don't want to call `cache()` after shuffle, as it can potentially make the dataset only get shuffled once.)
- This is one of the less tuned part of the codebase, I mostly just followed the documentation and its recommendations. You could probably get a better throughput by tinkering with the pipeline.

### Dynamic Batching?

One neat trick to increase the training speed in PyTorch is to group sequences with similar length together, pick a batch of them from this group, and then pad to the maximum length of that batch. This often greatly reduce the padding required and the average sequence length after padding.

However, this trick doesn't seem to work well when training with TPU. Fixed-size input tensors run fastest on TPU in my experience. Nonetheless, I'll describe how to do it below in case you are interested.

1. Remove padding in the TFRecord preparation script/function. (duh)
2. Use `tf.io.FixedLenSequenceFeature` in features_description (code below).
3. Replace `batch()` with `padded_batch(batch_size, padded_shapes=None)`.
4. Change the `tf.function` decorators to `tf.function(experimental_relax_shapes)` to allow input tensors of different shapes without retracing (i.e., recompiling).

```python
features_description = {
    "input_ids_question": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "input_mask_question": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "input_ids_answer": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "input_mask_answer": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "labels": tf.io.FixedLenFeature([30], tf.float32),
}
```

### Distributed Dataset

Distributed training with custom training loop requires converting a `tf.data.Dataset` instance into a distributed dataset:

```python
train_dist_ds = strategy.experimental_distribute_dataset(
    train_ds)
valid_dist_ds = strategy.experimental_distribute_dataset(
    valid_ds)
```

If you're using the Keras `fit()` API, you won't need to do this conversion.

## The Siamese Encoder Network

Finally, let's take a look at the neural network model that put tags on the input pair of question and answer. The tokenized question and answer is feed to the same RoBERTa encoder (a.k.a. Siamese network). The hidden states of the last layer of the encoder is put through an average pooling layer. Here's the code of the encoder([source code location](https://github.com/ceshine/kaggle-quest/blob/795d94c70f7c97fd2c0a5f383fdf52571f9bb0ed/quest/models.py#L38)):

(Reminder: we are using the RoBERTa model from [huggingface/transformers](https://github.com/huggingface/transformers) here.)

```python
class RobertaEncoder(TFRobertaPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.roberta = TFRobertaMainLayer(config, name="roberta")
        self.pooling = AveragePooling()

    def call(self, inputs, **kwargs):
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = tf.ones(
                tf.shape(inputs["input_ids"])[:2], tf.int32
            )
        outputs = self.roberta(inputs, **kwargs)[0]
        return self.pooling(outputs, inputs["attention_mask"])
```

There are 30 types of target labels, and I split them into three categories:

1. Those only related to the question
2. Those only related to the answer
3. Those requires information from both the question and the answer

We create three classification head for each category. Each head only uses data from the relevant encoder (e.g., the question head will only use results from the question encoder) to reduce over-fitting. I also found that putting a context gating on the intermediate states slightly improves the accuracy.

The top-level model code with some details omitted([source code location](https://github.com/ceshine/kaggle-quest/blob/795d94c70f7c97fd2c0a5f383fdf52571f9bb0ed/quest/models.py#L54)) :

```python
class DualRobertaModel(tf.keras.Model):
    def __init__(self, config, model_name, pretrained: bool = True):
        # omitted...
        if pretrained:
            self.roberta = RobertaEncoder.from_pretrained(
                model_name, config=config, name="roberta_question")
        else:
            self.roberta = RobertaEncoder(
                config=config, name="roberta_question")
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.q_classifier = tf.keras.layers.Dense(
            len(QUESTION_COLUMNS),
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            # omitted...
        )
        self.a_classifier = tf.keras.layers.Dense(
            len(ANSWER_COLUMNS),
            # omitted...
        )
        self.j_classifier = tf.keras.layers.Dense(
            len(JOINT_COLUMNS),
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            # omitted...
        )
        # omitted...

    # omitted...

    def call(self, inputs, **kwargs):
        pooled_output_question = self.roberta(
            {
                "input_ids": inputs["input_ids_question"],
                "attention_mask": inputs["attention_mask_question"]
            }, **kwargs
        )
        pooled_output_answer = self.roberta(
            {
                "input_ids": inputs["input_ids_answer"],
                "attention_mask": inputs["attention_mask_answer"]
            }, **kwargs
        )
        combined = tf.concat(
            [
                pooled_output_question, pooled_output_answer,
                pooled_output_answer * pooled_output_question
            ],
            axis=1
        )
        q_logit = self.q_classifier(self.dropout(
            self.gating_q(
                pooled_output_question
            ), training=kwargs.get("training", False)
        ))
        # omitted...
        logits = tf.concat(
            [q_logit, a_logit, j_logit],
            axis=1
        )
        # omitted...
```

PyTorch developer should find the above quite similar to a PyTorch model.

# Wrapping Up

In this post we briefly discuss how the learning curve of Tensorflow has significantly reduced in the 2.x release, and how the TPU has become more accessible than ever. We also present a case study of solving a Q&A labeling problem by fine-tuning RoBERTa-base model from _huggingface/transformer_ library, and with it some code snippets that could be useful for those who are more familiar with PyTorch.

In fact, Tensorflow 2.1 and TPU are also very good fit for CV applications. I already have another CV project in the pipeline and I haven't met any unsurmountable obstacles yet. I'll probably publish another case study of solving a image classification problem in about a month.

I'd love to hear from you. If you find any details in the presented codebase confusing, please let me know in the comment section. I'll add a section to this post or write a bonus post.
