---
slug: 16th-place-google-research-football
date: 2020-12-28T00:00:00.000Z
title: "[Kaggle] Google Research Football 2020"
description: "Describing my 16th place solution and also reviewing some of the others'"
tags:
  - pytorch
  - rl
  - kaggle
keywords:
  - reinforcement learning
  - pytorch
  - kaggle
url: /post/16th-place-google-research-football/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/ball-stadium-football-soccer-ball-488700/)" >}}

(This post an expansion of [this Kaggle post](https://www.kaggle.com/c/google-football/discussion/204666).)

## My Solution

Thanks to Kaggle, Manchester City F.C., and Google Research for this fantastic competition. Working on this competition was the most fun I've had for a while.

**The tl;dr version of my solution is that I used an MLP model to stochastically imitate WeKick's agents, with some rules to help it navigate in unfamiliar waters.**

### Why this Approach

After I got the GCP coupon, I looked at the competition timeline and thought that there is no way I can train a competitive RL agent from scratch in less than two weeks. I had to find some way to cut the training time shorter.

Then I found Ken Miller's [RL approximation notebook](https://www.kaggle.com/mlconsult/1149-ish-bot-rl-approximation) and learned that the imitation strategy works reasonably well. So I decided to use a similar approach to (pre)train NN models to bootstrap my policy network, with an episode scraper based on Felipe Bivort Haiek's [Google Football Episode Scraper quick fix](https://www.kaggle.com/felipebihaiek/google-football-episode-scraper-quick-fix). Huge thanks to both of you!

After a while, I found a lot of low-hanging fruits by tuning the features and assisting rules. As the model training is restricted to experience replays, no exploration was possible. The trained agent will not know how to act in a vast part of the state space, so setting some rules to guide them back to familiar territories can be quite helpful. I decided that tuning the imitator agents would be a better use of my time than an RL moonshot. Taking the safer bet paid off and gave me two agents with ~1,300 scores on the final day.

### More Details

- Episodes: Only episodes in which both agents were high-ranking were used. It was most likely a mistake. It probably made my agents perform poorly when playing against low-ranking agents. A better approach might be a submission-based filter.
- Features: Active-player-centered coordinates; coordinates and velocities of the goalkeepers are kept separately from the rest; distance-sorted coordinates, distances, angles, and velocities for the rest; Sprint and dribble sticky states; ball features; an offside feature(doesn't seem to work as expected).
- Stochastic actions: I reduced the temperature of the softmax function (by doubling the logits). This seems to work better than deterministic/greedy actions.
- Rules:
  - Using direction sticky states in features leads to overfitting and erratic behaviors, so I removed them. The downside is having the player high-passing the ball back to the goalie in free-kicks and resulting in own-goals. So I instructed the player to simply go right in free-kicks.
  - The model controls only the normal and throw-ins game modes. Training the model to play other modes seem to be a waste of time.
  - No dribbling.
  - Sprinting is entirely rule-based. I had found this to work slightly better in the final days as it doesn't waste precious time giving the sprint command in critical moments.
  - Turn the player back when they are taking the ball straight out of the field.
  - The defense is the most tricky part. In early experiments, the models would give absurd commands when the ball controlled by other players is too far from the active player (possibly due to the lack of exploration during training). Therefore, I kept the agents on a tight leash when defending. The models were only allowed to take over when the ball is near enough or in high-passes; the running-towards-the-ball strategy is used otherwise. This approach gives away many easy shot opportunities to the other teams. Only on the last day had I realized that the leash was too tight for the latest models. Loosen the leash gave better results in local tests. Unfortunately, I only had two submissions left at that point, so I could not properly tune this hyper-parameter.
- Local testing: At first, I let my agent play against the easy built-in AI and see if they can consistently beat the AI for more than five goals. Later I've found that the larger goal margin in easy mode doesn't translate to better performance against high-ranking agents, so I switch to the hard built-in AI.

### Code

My code is public on Github([ceshine/google-football-2020](https://github.com/ceshine/google-football-2020)). It includes

## Others' Solutions

To be updated...
