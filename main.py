import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)

state_size = 4
action_size = env.action_space.n
max_episodes = 10000
gamma = 0.95


def discount_and_normalize_rewards(episode_rewards):
    """
    iteratively compute the discounted rewards a each step of th episode given the later rewards
    :param episode_rewards: list of the rewards in the episode
    :return: discounted_episode_rewards: list of discounted rewards
    """
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    total_episode = 0.0
    for i in range(len(episode_rewards)):
        j = len(episode_rewards)-(i+1)
        total_episode = total_episode * gamma + episode_rewards[j]
        discounted_episode_rewards[j] = total_episode

    avg = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - avg) / (std)
    return discounted_episode_rewards


def play_episode(env, sess):
    episode_states, episode_actions, episode_rewards = [], [], []
    state = env.reset()
    # env.render()
    final = False
    while True:
        action_probabilities = sess.run(action_distribution, feed_dict={input_: state.reshape([1,4])})[0]
        action = np.random.choice(range(action_probabilities.shape[0]),
                                  p=action_probabilities)
        new_state, reward, final, info = env.step(action)
        episode_states += [state]
        episode_rewards += [reward]
        action_ = np.zeros(len(action_probabilities))
        action_[action] = 1
        episode_actions += [action_]
        state = new_state
        if final:
            break
    return episode_states, episode_actions, episode_rewards


episodeRewards = []
total_rewards = []
maximumRewardRecorded = 0

# simple fully connected NN with the special loss to match the REINFORCE policy gradient
with tf.name_scope("inputs"):
    input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
    actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
    discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards")

    # Add this placeholder for having this variable in tensorboard
    mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")

with tf.name_scope("fc1"):
    fc1 = tf.contrib.layers.fully_connected(inputs=input_,
                                            num_outputs=10,
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer())

with tf.name_scope("fc2"):
    fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                            num_outputs=5,
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer())

with tf.name_scope("fc3"):
    fc3 = tf.contrib.layers.fully_connected(inputs=fc2,
                                            num_outputs=action_size,
                                            activation_fn=None,
                                            weights_initializer=tf.contrib.layers.xavier_initializer())

with tf.name_scope("softmax"):
    action_distribution = tf.nn.softmax(fc3)

with tf.name_scope("loss"):
    neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=actions)
    loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_)

with tf.name_scope("train"):
    train_opt = tf.train.AdamOptimizer().minimize(loss)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    maxReward = 0
    for episode in range(max_episodes):
        state = env.reset()
        # env.render()
        episode_states, episode_actions, episode_rewards = play_episode(env, sess)
        episode_sum = np.sum(episode_rewards)
        total_rewards += [episode_sum]
        maxReward = max(maxReward, episode_sum)

        print("=====================================")
        print("Episode: ", episode)
        print("Reward: ", episode_sum)
        print("Max reward so far: ", maxReward)
        discountedRewards = discount_and_normalize_rewards(episode_rewards)
        loss_, _ = sess.run([loss, train_opt], feed_dict={input_: np.vstack(np.array(episode_states)),
                                                          actions: np.vstack(np.array(episode_actions)),
                                                          discounted_episode_rewards_: discountedRewards
                                                          })
    plt.plot(total_rewards)

    saver.save(sess, './%s_episodes.ckpt' %max_episodes)