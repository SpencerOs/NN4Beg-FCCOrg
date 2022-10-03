import tensorflow_probability as tfp # we are using a different module from tensorflow this time
import tensorflow as tf

tfd = tfp.distributions # making a shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.2,0.8]) # first day has 80% chance of being cold
transition_distribution = tfd.Categorical(probs=[[0.3,0.7],  # a cold day has 30% chance of being followed by a hot day
                                                 [0.2,0.8]]) # a hot day has 20% chance of being followed by a cold day
observation_distribution = tfd.Normal(loc=[0.,15.], scale=[5.,10.]) # On each day the temperature is normally distributed with mn/stdDev 0 and 5 on a cold day and mn/stdDev 15 and 10 on a hot day
# the loc argument represents the mean and the scale is the standard deviation

model = tfd.HiddenMarkovModel(
    initial_distribution = initial_distribution,
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps = 7
)

mean = model.mean()
# due to the way Tensorflow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor

# in the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:
    print(mean.numpy())