import pypownet.agent
import pypownet.environment

class Submission(pypownet.agent.Agent):
    """
    An agent which load the parameters of a multi-layer perceptron previously trained, 
    in order to predict the action to take given an observations.
    The MLP predict the id of the action, which is train transformed into a numpy array, and then into an Action object.
    """

    def __init__(self, environment):
        super().__init__(environment)

    def act(self, observation):
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        # Implement your policy here.
        do_nothing_action = self.environment.action_space.get_do_nothing_action()

        return do_nothing_action

        # No learning (i.e. self.feed_reward does pass)


