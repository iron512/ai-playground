# AI playground
I always took some time to develop various side projects, namely, the **sunday projects**. I stole the name from an old pal of mine and it kinda explains the concept, even tho most of them are not developed during any sunday (should I consider working on the *deep night projects*?)

I has been a solid year since I last took on a sunday project. I have been in a phase where I just wanted to stay logged if not at work. Now I kinda miss the whole dev for fun idea, which has been a consistent hobby i had during the uni years.

So, here I am, developing something vaguely bound to AI. I probabily will focus on simple Machine Learing models (for finding simple patterns on some data) or some reinforcement learning. I will try to keep some consistent track of my progress, but everybody knows that I wont. Cya.

## Progress
- In these many days it as been hard to find time to develop. But here I am, on the 1st of april 2025, with the Cartpole environment solved, a mild knowledge of the basic functioning of the PPO algorithm a serious will to continue improving on this path.

## TakeAway
Learning is process where I gather information, forget, recall, forget again and so on, until you gathered the same info as many times as possible and you don't have to look it up again. So I want to leave here some random notes to my future self and/or anyone reading this repository.

- The model *rollout_steps* are the count of steps done before updating the policy. Those steps are divided in random batches of *batch_size* steps. Standard values are 2048 and 64, meaning that there are going to be 32 different batches to update the policy model. This operation is executed on the same 2048 steps (but in different random combination) for the number of *epochs*. The randomness helps to break unwanted temporal patterns. Finally, the *total_timesteps* is the number of steps that are going to be analyzed before stopping the learning.
  - Smaller batches lead to **noisier, but with a better exploration**, while bigger batches lead to **more stable models, preferring experience to exploration**.
  - Bigger epoch size per rollout improve the learning, but lead to **overfitting**, while smaller epoch size emphasize the fresh update at the cost of learning. Smaller epoch size generally require viewing more steps
  
  - If the training is slow, *reduce n_steps or n_epoch*. If it doesn't improve, *increase the total_timesteps (and/or the n_steps)*. If unstable, grow the batch size