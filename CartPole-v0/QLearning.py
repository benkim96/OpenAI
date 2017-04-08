import random
import numpy as np
import gym
import matplotlib.pyplot as plt
import queue

env = gym.make('CartPole-v0')
GAMMA = 0.99
NUM_EPISODES = 1000
MIN_EXPLORE_RATE = 0.001
MIN_LEARNING_RATE = 0.1
Q = np.random.rand(162, env.action_space.n)
#Q = np.zeros((162, env.action_space.n))
Q = np.loadtxt('cp-data.txt')

learningRates = []
exploreRates = []
countRates = []

y = np.arange(NUM_EPISODES + 1)
q = queue.Queue(100)

def get_Box(obv):
	x, x_dot, theta, theta_dot = obv

	if x < -.8:
		box_number = 0
	elif x < .8:
		box_number = 1
	else:
		box_number = 2

	if x_dot < -.5:
		pass
	elif x_dot < .5:
		box_number += 3
	else:
		box_number += 6

	if theta < np.radians(-12):
		pass
	elif theta < np.radians(-1.5):
		box_number += 9
	elif theta < np.radians(0):
		box_number += 18
	elif theta < np.radians(1.5):
		box_number += 27
	elif theta < np.radians(12):
		box_number += 36
	else:
		box_number += 45

	if theta_dot < np.radians(-50):
		pass
	elif theta_dot < np.radians(50):
		box_number += 54
	else:
		box_number += 108

	return box_number

def update_explore_rate(episode, count100):
	newRate = max(MIN_EXPLORE_RATE, min(1.0, 1.0 - np.log10((episode+1)/10)))
	exploreRates.append(newRate*100)
	print("Explore Rate:", newRate)
	return newRate

def update_learning_rate(episode, count100):
	if count100 > 80:
		newRate = 1 - count100/100.0
	else:
		newRate = max(MIN_LEARNING_RATE, min(1.0, 1.0 - np.log10((episode+1)/10)))

	learningRates.append(newRate*100)
	print("Learning Rate:", newRate)
	return newRate

def update_action(state, explore_rate):
	if random.random() < explore_rate:
		return env.action_space.sample()
	else:
		return np.argmax(Q[state])

def q_learn():
	count100 = 0
	total_reward = 0
	total_completions = 0
	explore_rate = update_explore_rate(0, 0)
	learning_rate = update_learning_rate(0, 0)

	first_completion = 0
	completed = False

	for i in range(NUM_EPISODES):
		observation = env.reset()
		state_0 = get_Box(observation)
		steps = 0
		for _ in range(250):
			if (NUM_EPISODES - i) < 3:
				env.render()
			#env.render()
			action = update_action(state_0, explore_rate)
			obv, reward, done, info = env.step(action)

			state_1 = get_Box(obv)
			q_max = np.max(Q[state_0])
			Q[state_0, action] += learning_rate*(reward + GAMMA*np.amax(Q[state_1]) - Q[state_0, action])

			state_0 = state_1
			total_reward += reward

			steps = (_)

			if done:
				if q.full():
					count100 -= q.get()
				q.put(0)
				break

			if _ > 195:
				if q.full():
					count100 -= q.get()
				q.put(1)
				count100 +=  1
				total_completions += 1
				if not completed:
					first_completion = i
					completed = True
				break

		learning_rate = update_learning_rate(i, count100)
		explore_rate = update_explore_rate(i, count100)
		countRates.append(count100)
		print("Total: ", total_completions)
		print("Trial: ", i)
		print("100 score: ", count100)


	print("First Completion: ", first_completion)
	print("Completions/Total: ", total_completions/NUM_EPISODES)

def main():
	q_learn()
	countRates.append(100)
	plt.plot(y, learningRates, 'blue')
	plt.plot(y, exploreRates, 'green')
	plt.plot(y, countRates, 'red')
	plt.axis([0, NUM_EPISODES, 0, 101])
	plt.legend(['learning rate', 'explore rate', 'last 100 score'], loc='lower right')
	plt.show()

	np.savetxt('cp-data.txt', Q, fmt='%d')

if __name__ == "__main__":
	main()
