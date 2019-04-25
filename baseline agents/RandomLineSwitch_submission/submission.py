import pypownet.agent
import pypownet.environment

import numpy as np

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
        action_space = self.environment.action_space

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action()
        action_space.set_lines_status_switch_from_id(action=action,
                                                     line_id=np.random.randint(action_space.lines_status_subaction_length),
                                                     new_switch_value=1)



        return action


