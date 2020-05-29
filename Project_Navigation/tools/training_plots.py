import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCORE_ENVIRONMENT_SOLVED = 13

basic_dqn_mean_score_file = "data/run_BasicDQN-tag-Mean_Score.csv"
double_dqn_mean_score_file = "data/run_DoubleDQN-tag-Mean_Score.csv"
dueling_dqn_mean_score_file = "data/run_DuelingDQN-tag-Mean_Score.csv"
priority_replay_dqn_mean_score_file = "data/run_PriorityReplayDQN-tag-Mean_Score.csv"
double_dueling_dqn_mean_score_file = "data/run_DoubleAndDuelingDQN-tag-Mean_Score.csv"

basic_dqn_score_file = "data/run_BasicDQN-tag-Score.csv"
double_dqn_score_file = "data/run_DoubleDQN-tag-Score.csv"
dueling_dqn_score_file = "data/run_DuelingDQN-tag-Score.csv"
priority_replay_dqn_score_file = "data/run_PriorityReplayDQN-tag-Score.csv"
double_dueling_dqn_score_file = "data/run_DoubleAndDuelingDQN-tag-Score.csv"

basic_dqn_mean_score_df = pd.read_csv(basic_dqn_mean_score_file)
double_dqn_mean_score_df = pd.read_csv(double_dqn_mean_score_file)
dueling_dqn_mean_score_df = pd.read_csv(dueling_dqn_mean_score_file)
priority_replay_dqn_mean_score_df = pd.read_csv(priority_replay_dqn_mean_score_file)
double_dueling_dqn_mean_score_df = pd.read_csv(double_dueling_dqn_mean_score_file)

basic_dqn_score_df = pd.read_csv(basic_dqn_score_file)
double_dqn_score_df = pd.read_csv(double_dqn_score_file)
dueling_dqn_score_df = pd.read_csv(dueling_dqn_score_file)
priority_replay_dqn_score_df = pd.read_csv(priority_replay_dqn_score_file)
double_dueling_dqn_score_df = pd.read_csv(double_dueling_dqn_score_file)

f, (ax_scores, ax_mean_scores) = plt.subplots(1, 2)

ax_mean_scores.plot(basic_dqn_mean_score_df["Step"], basic_dqn_mean_score_df["Value"], label="Basic DQN")
ax_mean_scores.plot(double_dqn_mean_score_df["Step"], double_dqn_mean_score_df["Value"], label="Double DQN")
ax_mean_scores.plot(dueling_dqn_mean_score_df["Step"], dueling_dqn_mean_score_df["Value"], label="Dueling DQN")
ax_mean_scores.plot(priority_replay_dqn_mean_score_df["Step"], priority_replay_dqn_mean_score_df["Value"], label="DQN with Priority Experience Replay")
ax_mean_scores.plot(double_dueling_dqn_mean_score_df["Step"], double_dueling_dqn_mean_score_df["Value"], label="Double and Dueling DQN")
ax_mean_scores.axhline(SCORE_ENVIRONMENT_SOLVED, color="green", alpha=0.5)
ax_mean_scores.set_ylim([-2, 27])
ax_mean_scores.set_xlim([0, 1000])
ax_mean_scores.set_yticks(np.arange(-2, 27, step=2.0))
ax_mean_scores.set_xlabel("Episode")
ax_mean_scores.set_ylabel("Mean Score Over 100 Episodes")
ax_mean_scores.legend()

ax_scores.plot(basic_dqn_score_df["Step"], basic_dqn_score_df["Value"], label="Basic DQN")
ax_scores.plot(double_dqn_score_df["Step"], double_dqn_score_df["Value"], label="Double DQN")
ax_scores.plot(dueling_dqn_score_df["Step"], dueling_dqn_score_df["Value"], label="Dueling DQN")
ax_scores.plot(priority_replay_dqn_score_df["Step"], priority_replay_dqn_score_df["Value"], label="DQN with Priority Experience Replay")
ax_scores.plot(double_dueling_dqn_score_df["Step"], double_dueling_dqn_score_df["Value"], label="Double and Dueling DQN")
ax_scores.axhline(SCORE_ENVIRONMENT_SOLVED, color="green", alpha=0.5)
ax_scores.set_ylim([-2, 27])
ax_scores.set_xlim([0, 1000])
ax_scores.set_yticks(np.arange(-2, 27, step=2.0))
ax_scores.set_xlabel("Episode")
ax_scores.set_ylabel("Score per Episode")
ax_scores.legend()

plt.show()
