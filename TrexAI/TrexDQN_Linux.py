from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller
from gymnasium.spaces import Box, Discrete
from gymnasium import Env
import tkinter as tk
from mss import mss
import numpy as np
import pytesseract
import time
import cv2


class WebGame(Env):
    """
    A custom environment for interacting with a web game.
    """

    def __init__(self):
        """
        Initializes the WebGame environment with default parameters.
        """
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(1, 83, 100), dtype=np.uint8)
        self.action_space = Discrete(3)
        self.cap = mss()
        self.game_region = {'top': 0, 'left': 0, 'width': 0, 'height': 0}
        self.gameover_region = {'top': 0, 'left': 0, 'width': 0, 'height': 0}
        self.next_region = {'top': 0, 'left': 0, 'width': 0, 'height': 0}
        self.cum_reward = 0
        self.keyboard = Controller()
        self.mouse = MouseController()
        self.step_timer = time.time()

    def step(self, action):
        """
        Takes an action in the environment and returns the new observation, reward, done, and info.

        Args:
            action (int): The action to be taken.

        Returns:
            tuple: (new_observation, reward, done, done, info) where `done` is a boolean indicating if the episode is over.
        """
        action_map = {
            0: 'space',
            1: 'down',
            2: 'no_op'
        }
        if action_map[action] != 'no_op':
            if action_map[action] == 'space':
                self.keyboard.press(Key.space)
                self.keyboard.release(Key.space)
            elif action_map[action] == 'down':
                self.keyboard.press(Key.down)
                self.keyboard.release(Key.down)

        game_over_cap, game_over = self.gameover()
        new_observation = self.get_observation()
        reward = 1

        if action == 0 and self.get_next_observation():
            reward += 10
        elif action == 0 and not self.get_next_observation():
            reward -= 3

        if game_over:
            reward -= 10

        # Additional reward for longer runs
        if self.cum_reward >= 100:
            reward += 5
        if self.cum_reward >= 500:
            reward += 10

        self.cum_reward += reward
        info = {}
        print(f'Step timer:{time.time() - self.step_timer}s')
        self.step_timer = time.time()
        return new_observation, reward, game_over, game_over, info

    def render(self):
        """
        Renders the current state of the environment. This function is not implemented.
        """
        pass

    def close(self):
        """
        Closes any open windows and cleans up resources.
        """
        cv2.destroyAllWindows()

    def reset(self, **kwargs):
        """
        Resets the environment to its initial state and returns the initial observation.

        Returns:
            tuple: (initial_observation, {}) where `initial_observation` is the initial state of the environment.
        """
        self.mouse.click(Button.left, 1)
        self.keyboard.press(Key.space)
        self.keyboard.release(Key.space)
        print(f'cum reward for this session: {self.cum_reward}')
        self.cum_reward = 0
        return self.get_observation(), {}

    def gameover(self):
        """
        Checks if the game is over by analyzing the game over region.

        Returns:
            tuple: (cap, game_over) where `cap` is the screenshot of the game over region and `game_over` is a boolean indicating if the game is over.
        """
        cap = np.array(self.cap.grab(self.gameover_region))[:, :, :3]

        pytesseract.pytesseract.tesseract_cmd = '/path/to/tesseract'  # usually /usr/bin/tesseract
        text = pytesseract.image_to_string(cap)
        game_over = False
        for opt in ['GAME', 'GANE', 'OVER', 'GAHE']:
            if opt in text:
                game_over = True

        return cap, game_over

    def set_observation_region(self):
        """
        Opens a window to let the user select the observation region for the game.

        Returns:
            dict: The selected region's coordinates and dimensions.
        """

        def capture_game(event):
            def draw_rectangle(event, x, y, flags, params):
                """
                Draws a rectangle on the image based on mouse events.

                Args:
                    event (int): The type of mouse event.
                    x (int): The x-coordinate of the mouse event.
                    y (int): The y-coordinate of the mouse event.
                    flags (int): Additional flags for the event.
                    params: Additional parameters.
                """
                nonlocal drawing, rect_start, rect_end, edited_cap, game_region
                if event == cv2.EVENT_LBUTTONDOWN and not game_region:
                    rect_start = (x, y)
                    rect_end = (x, y)
                    drawing = True

                elif event == cv2.EVENT_MOUSEMOVE:
                    if drawing:
                        rect_end = (x, y)

                elif event == cv2.EVENT_LBUTTONUP and drawing:
                    drawing = False
                    game_region = (
                        rect_start[0], rect_start[1], rect_end[0] - rect_start[0], rect_end[1] - rect_start[1])
                    rect_end = (x, y)
                    cv2.rectangle(edited_cap, rect_start, rect_end, (0, 255, 0), 2)

            def re_selection(event):
                """
                Resets the game region selection.

                Args:
                    event (int): The type of event.
                """
                nonlocal game_region, edited_cap, cap
                game_region = None
                edited_cap = cap.copy()

            def save_selection(event):
                """
                Saves the selected game region and closes the window.

                Args:
                    event (int): The type of event.
                """
                nonlocal root, getting_observation, game_region
                getting_observation = False
                root.destroy()
                game_region = {'top': game_region[0], 'left': game_region[1], 'width': game_region[2],
                               'height': game_region[3]}

            nonlocal root, label, game_region
            cap = self.get_screenshot()
            getting_observation = True
            edited_cap = cap.copy()
            drawing = False
            rect_start = (0, 0)
            rect_end = (0, 0)

            cv2.namedWindow('Image')
            cv2.setMouseCallback('Image', draw_rectangle)
            while getting_observation:
                cap_copy = edited_cap.copy()

                # Draw the rectangle on the image copy
                if drawing:
                    cv2.rectangle(cap_copy, rect_start, rect_end, (0, 255, 0), 2)
                if game_region:
                    root.focus_force()
                    root.unbind('<Return>')
                    root.bind('<BackSpace>', re_selection)
                    root.bind('<space>', save_selection)
                    label.config(text='green rectangle should cover\n all of the game frame\n'
                                      'to approve press enter\n'
                                      'to cancel and rechoose press spacebar\n'
                                      f'{game_region}')
                    label.update()
                # Display the image with the rectangle
                cv2.imshow('Image', cap_copy)

                # Exit loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        game_region = None
        root = tk.Tk()
        root.title('Observation wizard')
        root.geometry('300x200')

        label = tk.Label(root, text="Place game screen on the location you want\n"
                                    "Press Enter to capture the game screenshot", padx=10, pady=10)
        label.pack(padx=20, pady=20)
        root.bind('<Return>', capture_game)

        root.mainloop()
        return game_region

    def get_screenshot(self):
        """
        Captures a screenshot of the entire monitor.

        Returns:
            np.ndarray: The screenshot as a NumPy array.
        """
        cap = np.array(self.cap.grab(self.cap.monitors[0]))
        return cap

    def get_observation(self):
        """
        Captures and preprocesses the observation from the game region.

        Returns:
            np.ndarray: The preprocessed observation.
        """
        raw_obs = np.array(self.cap.grab(self.game_region))
        gray_obs = cv2.cvtColor(raw_obs, cv2.COLOR_BGRA2GRAY)
        resized_obs = cv2.resize(gray_obs, (100, 83))

        observation = np.reshape(resized_obs, (1, 83, 100))
        return observation

    def get_next_observation(self):
        """
        Checks if the next observation indicates an obstacle.

        Returns:
            bool: True if the next observation shows an obstacle, otherwise False.
        """
        raw_obs = np.array(self.cap.grab(self.next_region))
        average_rgb = np.mean(raw_obs)
        if average_rgb < 250:
            return True
        else:
            return False


class CustomCheckpointCallback(BaseCallback):
    """
    A custom callback for saving model checkpoints during training.
    """

    def __init__(self, save_freq, save_path, verbose=0):
        """
        Initializes the checkpoint callback.

        Args:
            save_freq (int): Frequency of saving the model (in number of steps).
            save_path (str): Directory where the model checkpoints will be saved.
            verbose (int): Verbosity level.
        """
        super(CustomCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        """
        Called at each training step to handle checkpoint saving.

        Returns:
            bool: Always returns True to continue training.
        """
        # Save the model every `save_freq` steps
        if self.n_calls % self.save_freq == 0:
            model_path = f"{self.save_path}/best_{self.n_calls}"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saved model checkpoint to {model_path}")

        # Log additional information every 100 steps
        print(f"Step: {self.n_calls}, Reward: {self.locals['rewards']}, Done: {self.locals['dones']}")
        return True


if __name__ == '__main__':
    game = WebGame()
    # Linux setup
    game.game_region = {'left': 3, 'top': 490, 'width': 245, 'height': 117}
    game.gameover_region = {'left': 424, 'top': 376, 'width': 424, 'height': 54}
    game.next_region = {'left': 202, 'top': 556, 'width': 70, 'height': 106}

    # Setting window regions
    # game.set_observation_region()

    # Checking env
    # check_env(game)

    # Saved model path
    model_path = './models/DQN/best_model_name'  # Base folder /models/DQN/best_*

    # New model
    # model = DQN('CnnPolicy', game, verbose=1, learning_rate=0.0005, buffer_size=1200000, batch_size=64, exploration_fraction=0.2, learning_starts=2000, target_update_interval=500, train_freq=(4,'step'))  # Initialize DQN model

    # Load model
    model = DQN.load(model_path, game, verbose=1)

    checkpoint_callback = CustomCheckpointCallback(save_freq=5000, save_path='models/DQN', verbose=1)
    # Loading old policy:
    # old_model = DQN.load(model_path, game, verbose=1)
    # model.policy.load_state_dict(old_model.policy.state_dict())

    # Learn
    # model.learn(total_timesteps=100000, callback=checkpoint_callback)

    # Evaluate the model
    episodes = 10
    for episode in range(episodes):
        obs, _ = game.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = game.step(int(action))
            total_reward += reward
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
