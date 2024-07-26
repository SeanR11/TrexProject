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
    Custom environment for interacting with a web game.

    Inherits from gymnasium.Env and provides methods to step through the game,
    reset the environment, and capture game state.
    """

    def __init__(self):
        """
        Initializes the WebGame environment with observation and action spaces, and other attributes.
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
        Executes a step in the environment based on the given action.

        Args:
            action (int): The action to perform (0: space, 1: down, 2: no_op).

        Returns:
            tuple: New observation, reward, done (game over status), and additional info.
        """
        action_map = {
            0: 'space',
            1: 'down',
            2: 'no_op'
        }

        # Perform action
        if action_map[action] != 'no_op':
            if action_map[action] == 'space':
                self.keyboard.press(Key.space)
                self.keyboard.release(Key.space)
            elif action_map[action] == 'down':
                self.keyboard.press(Key.down)
                self.keyboard.release(Key.down)

        # Get game over status and new observation
        game_over_cap, game_over = self.gameover()
        new_observation = self.get_observation()

        # Calculate reward
        reward = 1
        if action == 0 and self.get_next_observation():
            reward += 6
        elif action == 0 and not self.get_next_observation():
            reward -= 3
        if game_over:
            reward -= 10
        if self.cum_reward >= 100:
            reward += 10
        if self.cum_reward >= 500:
            reward += 20

        self.cum_reward += reward
        info = {}
        print(f'Step timer: {time.time() - self.step_timer}s')
        self.step_timer = time.time()

        return new_observation, reward, game_over, game_over, info

    def render(self):
        """
        Optional method to render the environment. Not implemented in this case.
        """
        pass

    def close(self):
        """
        Closes the environment and releases any resources.
        """
        cv2.destroyAllWindows()

    def reset(self, **kwargs):
        """
        Resets the environment to its initial state.

        Returns:
            tuple: Initial observation and additional info.
        """
        self.mouse.click(Button.left, 1)
        self.keyboard.press(Key.space)
        self.keyboard.release(Key.space)
        print(f'cum reward for this session: {self.cum_reward}')
        self.cum_reward = 0
        time.sleep(1)
        return self.get_observation(), {}

    def gameover(self):
        """
        Checks if the game is over by analyzing the game over screen region.

        Returns:
            tuple: Captured image of the game over screen and game over status.
        """
        cap = np.array(self.cap.grab(self.gameover_region))[:, :, :3]
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update for Windows
        text = pytesseract.image_to_string(cap)
        game_over = any(opt in text for opt in ['GAME', 'GANE', 'OVER', 'GAHE'])
        return cap, game_over

    def set_observation_region(self):
        """
        Provides a GUI to select the region for capturing the game screen.

        Returns:
            dict: The region coordinates and dimensions for capturing the game.
        """

        def capture_game(event):
            def draw_rectangle(event, x, y, flags, params):
                """
                Draws a rectangle on the image to select the observation region.

                Args:
                    event: The event triggered.
                    x: The x-coordinate of the mouse event.
                    y: The y-coordinate of the mouse event.
                    flags: Flags for the event.
                    params: Additional parameters for the event.
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
                Resets the selection for the observation region.

                Args:
                    event: The event triggered.
                """
                nonlocal game_region, edited_cap, cap
                game_region = None
                edited_cap = cap.copy()

            def save_selection(event):
                """
                Saves the selected region and closes the GUI.

                Args:
                    event: The event triggered.
                """
                nonlocal root, getting_observation, game_region
                getting_observation = False
                root.destroy()
                game_region = {'top': game_region[1], 'left': game_region[0], 'width': game_region[2],
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
                cv2.imshow('Image', cap_copy)
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
        Captures a screenshot of the entire screen.

        Returns:
            np.ndarray: The captured screenshot.
        """
        cap = np.array(self.cap.grab(self.cap.monitors[0]))
        return cap

    def get_observation(self):
        """
        Captures and processes the current observation of the game.

        Returns:
            np.ndarray: The processed observation.
        """
        raw_obs = np.array(self.cap.grab(self.game_region))
        gray_obs = cv2.cvtColor(raw_obs, cv2.COLOR_BGRA2GRAY)
        resized_obs = cv2.resize(gray_obs, (100, 83))
        observation = np.reshape(resized_obs, (1, 83, 100))
        return observation

    def get_next_observation(self):
        """
        Checks if the next observation indicates a change.

        Returns:
            bool: True if the next observation indicates a change, otherwise False.
        """
        raw_obs = np.array(self.cap.grab(self.next_region))
        average_rgb = np.mean(raw_obs)
        return average_rgb < 250


class Checkpointcallback(BaseCallback):
    """
    Custom callback to save model checkpoints during training.
    """

    def __init__(self, save_freq, save_path, verbose=0):
        """
        Initializes the callback.

        Args:
            save_freq (int): Frequency of saving model checkpoints.
            save_path (str): Path to save the checkpoints.
            verbose (int): Verbosity level.
        """
        super(Checkpointcallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        """
        Called at each step to potentially save the model and log information.

        Returns:
            bool: Always returns True.
        """
        if self.n_calls % self.save_freq == 0:
            model_path = f"{self.save_path}/dqn_training_{self.n_calls}"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saved model checkpoint to {model_path}")

        print(f"Step: {self.n_calls}, Reward: {self.locals['rewards']}, Done: {self.locals['dones']}")
        return True


if __name__ == '__main__':
    def compare_weights(weights_1, weights_2):
        """
        Compares two sets of model weights and calculates the total difference.

        Args:
            weights_1 (dict): First set of weights.
            weights_2 (dict): Second set of weights.

        Returns:
            float: Total difference between the two sets of weights.
        """
        total_difference = 0.0
        for key in weights_1.keys():
            weight_diff = np.linalg.norm(weights_1[key].cpu().numpy() - weights_2[key].cpu().numpy())
            total_difference += weight_diff
            print(f"Weights for {key}: Difference = {weight_diff}")

        return total_difference


    # Initialize the game environment
    game = WebGame()

    # Define the regions for game capture (for Windows)
    game.game_region = {'left': 0, 'top': 500, 'width': 320, 'height': 150}
    game.gameover_region = {'left': 640, 'top': 354, 'width': 646, 'height': 68}
    game.next_region = {'left': 200, 'top': 556, 'width': 120, 'height': 95}

    # Uncomment this line to manually set the observation region
    # game.set_observation_region()

    # Check if the environment is working correctly
    check_env(game)

    # Initialize and train the model
    model = DQN('CnnPolicy', game, verbose=1,
                learning_rate=0.0005,
                buffer_size=1200000,
                batch_size=64,
                exploration_fraction=0.2,
                learning_starts=2000,
                target_update_interval=500,
                train_freq=(4, 'step'))

    # Uncomment this line to load a pre-trained model
    # model = DQN.load(model_path, game, verbose=1)

    # Uncomment this to load old policy and copy weights
    # old_model = DQN.load(model_path, game, verbose=1)
    # model.policy.load_state_dict(old_model.policy.state_dict())

    # Initialize checkpoint callback
    # checkpoint_callback = CustomCheckpointCallback(save_freq=5000, save_path='./models/DQN', verbose=1)

    # Train the model
    model.learn(total_timesteps=200000, callback=Checkpointcallback)

    # Evaluate the model
    episodes = 100
    obs = None
    for episode in range(episodes):
        obs, _ = game.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = game.step(int(action))
            total_reward += reward
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # Uncomment this section to test the regions
    # time.sleep(2)
    # cap = np.array(game.cap.grab(game.cap.monitors[0]))
    # cv2.rectangle(cap, (game.game_region['left'], game.game_region['top']),
    #               (game.game_region['left'] + game.game_region['width'],
    #                game.game_region['height'] + game.game_region['top']),
    #               (255, 0, 0), 1)
    # cv2.rectangle(cap, (game.next_region['left'], game.next_region['top']),
    #               (game.next_region['left'] + game.next_region['width'],
    #                game.next_region['height'] + game.next_region['top']),
    #               (0, 255, 0), 2)
    # cv2.rectangle(cap, (game.gameover_region['left'], game.gameover_region['top']),
    #               (game.gameover_region['left'] + game.gameover_region['width'],
    #                game.gameover_region['height'] + game.gameover_region['top']),
    #               (0, 0, 255), 3)
    # while True:
    #     cv2.imshow('test', cap)
    #     cv2.waitKey(1)