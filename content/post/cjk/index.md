---
slug: cjk-unicode
date: 2019-04-24T00:30:57.007Z
title: "Detecting Chinese Characters in Unicode Strings"
description: "CJK in Unicode and Python 3 implementation"
tags:
  - nlp
url: /post/cjk-unicode/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/peach-blossom-landscape-spring-4119429/)" >}}

# Motivation

I have a situation where an automatic mechanism to remove texts in a dataset that are not in Chinese. The dataset contains characters from Traditional Chinese, Simplified Chinese, English, and on some rare occasion French, Arabic, and other languages.

General purpose language detection packages (such as [this one](https://pypi.org/project/langdetect/)) produces a lot more false positives than expected. Texts with Chinese characters mixed with Latin characters are often classified as different languages. And quite often Chinese texts are classified as Korean, which is very interesting because the dataset does not have any Korean characters.

Since the tasks only requires a binary label (Chinese or not Chinese) for each input, I figure a better approach might be building my own algorithm that utilize the block range information of Unicode.

This post documents what I've learned about CJK characters in Unicode, and presents a better but still inperfect solution.

# CJK characters in Unicode

> It is a commonly used acronym for "Chinese, Japanese, and Korean". The term "CJK character" generally refers to "Chinese characters", or more specifically, the Chinese (= Han) ideographs used in the writing systems of the Chinese and Japanese languages, occasionally for Korean, and historically in Vietnam. [1]

In Unicode, the shared characters among Chinese, Japanese, and Korean, were identified and merged as "CJK Unified Ideographs". They includes characters used in Chinese writing system, *kanji* in Japanese, and *hanja* in Korean. [2]

So can we take a character out from the unified ideographs and tell if it is a Chinese, Japanese, or Korean characters? Sadly, the answer is no:

> It's basically impossible and largely meaningless. It's the equivalent of asking if "a" is an English letter or a French one. There are some characters where one can guess based on the source information in the Unihan Database that it's traditional Chinese, simplified Chinese, Japanese, Korean, or Vietnamese, but there are too many exceptions to make this really reliable.[1]

The good news is that Korean rarely use Chinese characters in modern days, and Japanese texts in most cases contains their own "hiragana" and "katakana" characters. The other cases with all Chinese characters, unfortunately, might require us to rely on probabilistic models that use language features. Distinguishing between Traditional and Simplified Chinese can be also quite difficult as well, and awaits further research.

## Respective Unicode Blocks

The "common" block of CJK Unified Ideographs should cover most of the case. We should be able to ignore the extensions when classifying.

|                              Block |     Range |  Comment |
|-----------------------------------:|----------:|---------:|
|             CJK Unified Ideographs | 4E00-9FFF | Common   |
|                           Hiragana | 3040-309F | Japanese [3] |
|                           Katakana | 30A0-30FF | Japanese [3] |
|                  Hangul Syllables  | AC00-D7A3 | Korean [4] |
| CJK Unified Ideographs Extension A | 3400-4DBF | Rare [5] |
| CJK Unified Ideographs Extension B | 20000-2A6DF | Rare, historic |
| CJK Unified Ideographs Extension C | 2A700–2B73F | Rare, historic |
| CJK Unified Ideographs Extension D | 2B740–2B81F | Uncommon, some in current use |
| CJK Unified Ideographs Extension E | 2B820–2CEAF | Rare, historic |
| CJK Compatibility Ideographs | F900-FAFF | Duplicates, unifiable variants, corporate characters |
| CJK Compatibility Ideographs Supplement | 2F800-2FA1F | Unifiable variants |

## Python 3 Implementation

I emphasize using Python **3** to simplify things. The default encoding for Python 3 source code is UTF-8, and the language's *str* type contains Unicode characters, meaning any string created using "unicode rocks!", 'unicode rocks!', or the triple-quoted string syntax is stored as Unicode [6].

The following implementation uses `re.search` to search for characters in the specified block(s). Some simple test cases are supplied, along with some incorrect cases, including Japanese texts with only Chinese characters, and a bizarre but common usage of a Japanese character in Tradtional Chinese texts.

{{< gist ceshine 1f8cd81ce34d89f1429d0928c28d97e4 >}}

# References

1. [[Unicode.org] FAQ - Chinese and Japanese](http://www.unicode.org/faq/han_cjk.html)
1. [[Wikipedia] CJK Unified Ideographs](https://www.wikiwand.com/en/CJK_Unified_Ideographs)
1. [[key-shortcut.com] Japanese](https://www.key-shortcut.com/en/writing-systems/%E3%81%B2%E3%82%89%E3%81%8C%E3%81%AA-japanese/)
1. [[Wikipedia] Hangul Syllables](https://www.wikiwand.com/en/Hangul_Syllables)
1. [[StackOverflow] What's the complete range for Chinese characters in Unicode?](https://stackoverflow.com/questions/1366068/whats-the-complete-range-for-chinese-characters-in-unicode)
1. [[python.org] Unicode HOWTO](https://docs.python.org/3.7/howto/unicode.html)