import os
import gym
import numpy as np


def done_grading(env):
    if hasattr(env, 'done_grading'):
        return env.done_grading()
    if hasattr(env, 'env'):
        return done_grading(env.env)
    if hasattr(env, '_env'):
        return done_grading(env._env)

def is_grading(env):
    if hasattr(env, 'is_grading'):
        return env.is_grading()
    if hasattr(env, 'env'):
        return is_grading(env.env)
    if hasattr(env, '_env'):
        return is_grading(env._env)



class RenderObservations(gym.Wrapper):
    def __init__(self, env, display_vector_obs=True):
        gym.Wrapper.__init__(self, env)
        self.viewer = None
        self._empty = np.zeros((1,1,1))
        self._has_vector_obs = hasattr(self.observation_space, 'spaces')
        self._8bit = None
        self._display_vector_obs = display_vector_obs
        
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        should_render = True
        if 'human_agent_display' in globals():
            global human_agent_display
            should_render = human_agent_display
        self._renderObs(ob, should_render)
        return ob, reward, done, info   

    def _renderObs(self, obs, should_render):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        if not should_render:
            self.viewer.imshow(self._empty)
            return self.viewer.isopen
        if self._has_vector_obs:
            visual_obs = obs['visual'].copy()
            vector_obs = obs['vector'].copy()
        else:
            visual_obs = obs.copy()           
        if self._has_vector_obs and self._display_vector_obs:
            w = 84
            # Displays time left and number of keys on visual observation
            key = vector_obs[0:-1]
            time_num = vector_obs[-1]
            key_num = np.argmax(key, axis=0)
            # max_bright = 1
            max_bright = 255
            visual_obs[0:10, :, :] = 0
            for i in range(key_num):
                start = int(i * 16.8) + 4
                end = start + 10
                visual_obs[1:5, start:end, 0:2] = max_bright
            visual_obs[6:10, 0:int(time_num * w), 1] = max_bright    
        self._8bit = visual_obs
        # if type(visual_obs[0][0][0]) is np.float32 or type(visual_obs[0][0][0]) is np.float64:
            # _8bit = (255.0 * visual_obs).astype(np.uint8)
        self._8bit = ( visual_obs).astype(np.uint8)
        self.viewer.imshow(self._8bit)
        return self.viewer.isopen

    def render(self, mode='human', **kwargs):
        if self.viewer:
            self.viewer.imshow(self._8bit)
        return self._8bit

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

