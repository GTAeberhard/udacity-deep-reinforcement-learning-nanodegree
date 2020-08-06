import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCORE_ENVIRONMENT_SOLVED = 30

ddpg_score_file = "data/run_DDPG-tag-Score.csv"
ddpg_mean_score_file = "data/run_DDPG-tag-Mean_Score.csv"
ddpg_actor_loss_file = "data/run_DDPG-tag-Actor_Loss.csv"
ddpg_critic_loss_file = "data/run_DDPG-tag-Critic_Loss.csv"

ddpg_score_df = pd.read_csv(ddpg_score_file)
ddpg_mean_score_df = pd.read_csv(ddpg_mean_score_file)
ddpg_actor_loss_df = pd.read_csv(ddpg_actor_loss_file)
ddpg_critic_loss_df = pd.read_csv(ddpg_critic_loss_file)

f, ((ax_scores, ax_mean_scores), (ax_actor_loss, ax_critic_loss)) = plt.subplots(2, 2)

# Mean Score
ax_mean_scores.plot(ddpg_mean_score_df["Step"], ddpg_mean_score_df["Value"], label="Deep Deterministic Policy Gradient (DDPG)")
ax_mean_scores.axhline(SCORE_ENVIRONMENT_SOLVED, color="green", alpha=0.5)
ax_mean_scores.set_ylim([-5, 100])
ax_mean_scores.set_xlim([0, 500])
ax_mean_scores.set_yticks(np.arange(-10, 100, step=10.0))
ax_mean_scores.set_xlabel("Episode")
ax_mean_scores.set_ylabel("Mean Score Over 100 Episodes")
ax_mean_scores.legend()

# Score
ax_scores.plot(ddpg_score_df["Step"], ddpg_score_df["Value"], label="Deep Deterministic Policy Gradient (DDPG)")
ax_scores.axhline(SCORE_ENVIRONMENT_SOLVED, color="green", alpha=0.5)
ax_scores.set_ylim([-5, 100])
ax_scores.set_xlim([0, 500])
ax_scores.set_yticks(np.arange(-10, 100, step=10.0))
ax_scores.set_xlabel("Episode")
ax_scores.set_ylabel("Score per Episode")
ax_scores.legend()

# Actor Loss
ax_actor_loss.plot(ddpg_actor_loss_df["Step"], ddpg_actor_loss_df["Value"],
                   label="Deep Deterministic Policy Gradient (DDPG)")
ax_actor_loss.set_ylim([-7, 1])
ax_actor_loss.set_xlim([0, 500])
ax_actor_loss.set_yticks(np.arange(-7, 1, step=1.0))
ax_actor_loss.set_xlabel("Episode")
ax_actor_loss.set_ylabel("Actor Loss")
ax_actor_loss.legend()

# Critic Loss
ax_critic_loss.plot(ddpg_critic_loss_df["Step"], ddpg_critic_loss_df["Value"],
                    label="Deep Deterministic Policy Gradient (DDPG)")
ax_critic_loss.set_ylim([0, 0.2])
ax_critic_loss.set_xlim([0, 500])
ax_critic_loss.set_yticks(np.arange(0, 0.2, step=0.05))
ax_critic_loss.set_xlabel("Episode")
ax_critic_loss.set_ylabel("Critic Loss")
ax_critic_loss.legend()

plt.show()
