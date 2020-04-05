import unittest
import torch
import logging
from BCQ_for_RSPMN import generate_buffer_from_dataset
import utils


class TestBCQ_for_RSPMN(unittest.TestCase):
    def setUp(self):
        pass


    def test_init(self):
        pass

    def test_generate_data_5rows_3timeSteps(self):

        data = self.simulation.generate_data(5, 3)
        self.assertEqual((5, 3, 3), data.shape, msg='data shape should be (5, 3, 3)'
                                                      ' instead it is {}'.format(data.shape))

    def test_generate_buffer_from_dataset(self):

        import numpy as np
        dataset = np.arange(150).reshape(10, 15)
        print(dataset)
        vars_in_single_time_step = 5
        sequence_length = 3
        state_vars = 2
        action_vars = 2
        state_dim = 2
        action_dim = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generate_buffer_from_dataset(dataset, vars_in_single_time_step,
                                     sequence_length, state_vars, action_vars,
                                     state_dim, action_dim, device)

        buffer_name = 'testBuffer'

        rows = dataset.shape[0]
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device,
                                           max_size=rows * (sequence_length-1))
        replay_buffer.size = 21
        replay_buffer.ptr = 20
        replay_buffer.load(f"./buffers/{buffer_name}")
        # save_folder = f"./buffers/{buffer_name}"
        # print(np.load("./buffers/testBuffer_state.npy")[:-1])
        # replay_buffer.state[:replay_buffer.size] = np.load("./buffers/testBuffer_state.npy")[:replay_buffer.size]
        print(f"state: {replay_buffer.state}")
        print(f"action: {replay_buffer.action}")
        print(f"reward: {replay_buffer.reward}")
        print(f"next_state: {replay_buffer.next_state}")
        print(f"not_done: {replay_buffer.not_done}")
        # data = slice_data_by_scope(data, data_scope, slice_scope)
        # print(data)