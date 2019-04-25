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

        # Select a random substation ID on which to perform node-splitting
        target_substation_id = np.random.choice(action_space.substations_ids)
        expected_target_configuration_size = action_space.get_number_elements_of_substation(target_substation_id)
        # Choses a new switch configuration (binary array)
        target_configuration = np.random.choice([0, 1], size=(expected_target_configuration_size,))
        
        # We modify the action
        action_space.set_switches_configuration_of_substation(action=action,
                                                              substation_id=target_substation_id,
                                                              new_configuration=target_configuration)

        # Ensure changes have been done on action
        current_configuration, _ = action_space.get_switches_configuration_of_substation(action, target_substation_id)
        assert np.all(current_configuration == target_configuration)


        return action

