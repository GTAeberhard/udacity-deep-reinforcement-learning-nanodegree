import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCORE_ENVIRONMENT_SOLVED = 0.5
NUM_EPISODES = 2200

score_file = "data/run_MADDPG-tag-Score.csv"
mean_score_file = "data/run_MADDPG-tag-Mean_Score.csv"
actor_loss_file = "data/run_MADDPG-tag-Actor_Loss.csv"
critic_loss_file = "data/run_MADDPG-tag-Critic_Loss.csv"
score_df = pd.read_csv(score_file)
mean_score_df = pd.read_csv(mean_score_file)
actor_loss_df = pd.read_csv(actor_loss_file)
critic_loss_df = pd.read_csv(critic_loss_file)

f, ((ax_scores, ax_mean_scores), (ax_actor_loss, ax_critic_loss)) = plt.subplots(2, 2)

f.suptitle("Multi-Agent Deep Deterministic Policy Gradient (MADDPG) in Tennis Environment")

# Mean Score
ax_mean_scores.plot(mean_score_df["Step"], mean_score_df["Value"])
ax_mean_scores.axhline(SCORE_ENVIRONMENT_SOLVED, color="green", alpha=0.5)
ax_mean_scores.set_ylim([0, 1.5])
ax_mean_scores.set_xlim([0, NUM_EPISODES])
ax_mean_scores.set_yticks(np.arange(0, 1.5, step=0.5))
ax_mean_scores.set_xlabel("Episode")
ax_mean_scores.set_ylabel("Mean Score Over 100 Episodes")

# Score
ax_scores.plot(score_df["Step"], score_df["Value"])
ax_scores.axhline(SCORE_ENVIRONMENT_SOLVED, color="green", alpha=0.5)
ax_scores.set_ylim([0, 3])
ax_scores.set_xlim([0, NUM_EPISODES])
ax_scores.set_yticks(np.arange(0, 3, step=0.5))
ax_scores.set_xlabel("Episode")
ax_scores.set_ylabel("Score per Episode")

# Actor Loss
ax_actor_loss.plot(actor_loss_df["Step"], actor_loss_df["Value"])
ax_actor_loss.set_ylim([-160, 10])
ax_actor_loss.set_xlim([0, NUM_EPISODES])
ax_actor_loss.set_yticks(np.arange(-160, 10, step=20))
ax_actor_loss.set_xlabel("Episode")
ax_actor_loss.set_ylabel("Actor Loss")

# Critic Loss
ax_critic_loss.plot(critic_loss_df["Step"], critic_loss_df["Value"])
ax_critic_loss.set_ylim([0, 200])
ax_critic_loss.set_xlim([0, NUM_EPISODES])
ax_critic_loss.set_yticks(np.arange(0, 200, step=20))
ax_critic_loss.set_xlabel("Episode")
ax_critic_loss.set_ylabel("Critic Loss")

plt.show()
