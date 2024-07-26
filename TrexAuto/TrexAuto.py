from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller
import pytesseract
import tkinter as tk
import numpy as np
import pytesseract
import threading
import time
import mss
import cv2


class TrexAuto:
    def __init__(self):
        """
        Initializes the TrexAuto class with necessary attributes and settings.
        """
        self.cap = mss.mss()  # Screen capture object
        self.obstacle_area = {'left': 180, 'top': 558, 'width': 220, 'height': 90}  # Area for detecting obstacles
        self.clear_area = {'left': 6, 'top': 580, 'width': 155, 'height': 80}  # Area for clearing obstacles
        self.game_over_area = {'left': 651, 'top': 369, 'width': 620, 'height': 44}  # Area for detecting game over
        self.bird_area = {'left': 139, 'top': 500, 'width': 224, 'height': 32}  # Area for detecting birds
        self.daytime_area = {'left': 50, 'top': 100, 'width': 10, 'height': 10}  # Area for detecting daytime
        self.score_area = {'left': 1707, 'top': 258, 'width': 175, 'height': 53}  # Area for detecting score
        self.environment = 'day'  # Initial environment setting
        self.keyboard = Controller()  # Keyboard controller
        self.mouse = MouseController()  # Mouse controller
        self.score = 0  # Initial score
        self.counter_thread = None  # Thread for updating score and areas
        self.main_loop = True  # Flag to control the main loop

    def run(self, window):
        """
        Main loop for running the auto bot.
        """
        self.jumping = False
        self.ducking = False
        self.state = None
        self._start_counter()  # Start the score counter thread
        while self.main_loop:
            # Capture the screen state
            self.state = np.array(self.cap.grab(self.cap.monitors[0]))

            # Check if the environment is day or night
            if self.check_area(self.state, self.daytime_area, 225):
                self.environment = 'night'
            else:
                self.environment = 'day'

            # Handle game over state
            if self.check_area(self.state, self.game_over_area, 230 if self.environment == 'day' else 80,
                               mode=self.environment):
                self.game_reset()

            # Handle bird detection and jumping
            if self.check_area(self.state, self.bird_area, 250 if self.environment == 'day' else 68,
                               mode=self.environment):
                self.keyboard.press(Key.down)
                time.sleep(0.3)
                self.keyboard.release(Key.down)

            # Handle obstacle detection and jumping
            elif self.check_area(self.state, self.obstacle_area, 254 if self.environment == 'day' else 64,
                                 mode=self.environment):
                self.keyboard.press(Key.space)

            # Handle clearing areas
            elif self.check_area(self.state, self.clear_area, 254 if self.environment == 'day' else 63.8,
                                 mode='night' if self.environment == 'day' else 'day'):
                self.keyboard.press(Key.down)
                time.sleep(0.06)
                self.keyboard.release(Key.down)

            # Update the GUI window
            window.update()

    def game_reset(self):
        """
        Resets the game state after a game over.
        """
        self.mouse.position = (self.game_over_area['left'], self.game_over_area['top'])
        self.obstacle_area = {'left': 180, 'top': 558, 'width': 220, 'height': 90}  # Reset obstacle area
        self.mouse.click(Button.left, 1)  # Click to restart the game
        self.keyboard.press(Key.space)  # Press space to start the game
        self.score = 0  # Reset score
        time.sleep(1)  # Wait for a second

    def _start_counter(self):
        """
        Starts a background thread to update the score and obstacle areas.
        """

        def counter_function():
            while self.main_loop:
                if self.state is not None:
                    current_score = self.get_score()  # Get the current score
                    if self.score < current_score:
                        self.score += 100
                        self.obstacle_area['width'] += 10
                        self.bird_area['width'] += 10
                        time.sleep(0.5)

        self.counter_thread = threading.Thread(target=counter_function)
        self.counter_thread.daemon = True
        self.counter_thread.start()

    def check_area(self, state, area, value, mode=None):
        """
        Checks if the specified area of the screen meets the criteria.
        """
        raw_ss = state[area['top']:area['top'] + area['height'], area['left']:area['left'] + area['width']]
        average_rgb = np.mean(raw_ss)
        if mode == 'night':
            return average_rgb > value
        return average_rgb < value

    def test_area(self, area, screen):
        """
        Tests the average RGB value in the specified area of the screen.
        """
        raw_ss = screen[area['top']:area['top'] + area['height'], area['left']:area['left'] + area['width']]
        average_rgb = np.mean(raw_ss)
        return average_rgb

    def get_score(self):
        """
        Retrieves the current score from the score area using OCR.
        """
        raw_ss = self.state[self.score_area['top']:self.score_area['top'] + self.score_area['height'],
                 self.score_area['left']:self.score_area['left'] + self.score_area['width']]
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        text = pytesseract.image_to_string(raw_ss, config='--psm 6 digits')
        clean_text = text.strip()
        if clean_text and clean_text.isdigit():
            if int(text) % 100 == 0:
                return int(text)
        return -1

    def check_status(self):
        """
        Checks if all required areas are set up.
        """
        flag = 0
        if self.obstacle_area and self.score_area and self.clear_area and self.bird_area and self.daytime_area:
            flag = 1
        self.main_loop = flag
        return flag

    def stop(self):
        """
        Stops the main loop.
        """
        self.main_loop = False


class GUI:
    def __init__(self):
        """
        Initializes the GUI for the TrexAuto application.
        """
        self.trex = TrexAuto()
        self.gui()

    def gui(self):
        """
        Sets up and displays the main GUI window.
        """
        self.window = tk.Tk()
        self.window.title('TrexManager')

        self.title = tk.Label(self.window, text='TrexManager', font='Arial', padx=50)
        self.title.pack(padx=0, pady=20)

        self.message = tk.Label(self.window, text='', font=('Arial', 12))
        self.message.place(x=25, y=50)

        self.setup_button = tk.Button(self.window, text='Setup Environment', command=self.set_environment)
        self.setup_button.pack(side='left', padx=20, pady=20)

        self.play_button = tk.Button(self.window, text='Start Play', command=self.play)
        self.play_button.pack(side='right', padx=20)

        self.preview_button = tk.Button(self.window, text='Show Preview', command=self.show_area_preview)
        self.preview_button.pack(side='left', padx=20)
        self.window.mainloop()

    def set_environment(self):
        """
        Starts the environment setup wizard.
        """
        self.drawing = False
        self.standby = False
        self.setup = False
        self.selection_start = (0, 0)
        self.selection_end = (0, 0)
        self.cap = mss.mss()
        self.stage = 'obstacle_area'

        self.sub_window = tk.Tk()
        self.sub_window.focus_force()

        self.sub_title = tk.Label(self.sub_window, text='Press enter to capture game screen\n'
                                                        'Press q to close wizard', font=('Arial', 17),
                                  padx=20, pady=20)
        self.sub_title.pack()

        self.sub_text = tk.Label(self.sub_window, text='', font=('Arial', 13), padx=20, pady=30)
        self.sub_text.pack()

        self.sub_window.bind('<Return>', lambda _: self.set_obstacle_area())
        self.sub_window.bind('<q>', lambda _: self.sub_window.destroy())

    def wizard_screen(self):
        """
        Displays the wizard screen for selecting areas.
        """
        while self.wizard_screen_active:
            self.window.update()
            if self.drawing or self.standby:
                cv2.rectangle(self.screen, self.selection_start, self.selection_end, (0, 255, 0), 1)
            cv2.imshow('screen', self.screen)
            self.screen = self.screen_copy.copy()
            key = cv2.waitKey(1)
            if key == 13:
                if self.stage == 'obstacle_area':
                    self.trex.obstacle_area = self.get_region()
                    self.set_clear_area()
                elif self.stage == 'clear_area':
                    self.trex.clear_area = self.get_region()
                    self.set_game_over_area()
                elif self.stage == 'game_over_area':
                    self.trex.game_over_area = self.get_region()
                    self.set_bird_area()
                elif self.stage == 'bird_area':
                    self.trex.bird_area = self.get_region()
                    self.set_daytime_area()
                elif self.stage == 'daytime_area':
                    self.trex.daytime_area = self.get_region()
                    self.set_score_area()
                elif self.stage == 'score_area':
                    self.trex.score_area = self.get_region()
                    self.wizard_screen_active = False
                    self.setup = True
            elif key == ord('q'):
                self.wizard_screen_active = False

        cv2.destroyAllWindows()
        self.sub_window.destroy()
        self.window.deiconify()
        if self.setup:
            self.message.config(text='Environment successfully set', foreground='#00ee00')
        else:
            self.message.config(text='Environment wizard closed', foreground='#ff0000')
        self.message.place(x=100, y=50)

    def set_obstacle_area(self):
        """
        Sets up the area for detecting obstacles.
        """
        self.sub_title.config(text='Obstacle Area')
        self.sub_text.config(text='Select with your mouse score detection area\n'
                                  'click q to close wizard')
        self.sub_text.update()
        self.sub_title.update()

        self.sub_window.unbind('<Return>')

        self.window.iconify()
        self.sub_window.iconify()
        time.sleep(0.2)
        self.screen = np.array(self.cap.grab(self.cap.monitors[0]))
        self.screen_copy = self.screen.copy()

        cv2.namedWindow('screen')
        cv2.setMouseCallback('screen', self.mouse_callback)

        self.sub_window.deiconify()
        self.wizard_screen_active = True
        self.wizard_screen()

    def set_clear_area(self):
        """
        Sets up the area for clearing obstacles.
        """
        self.standby = False
        self.stage = 'clear_area'
        self.sub_title.config(text='Clear Area')
        self.sub_text.config(text='Select with your mouse score detection area\n'
                                  'click q to close wizard')
        self.sub_text.update()
        self.sub_title.update()

    def set_game_over_area(self):
        """
        Sets up the area for detecting game over.
        """
        self.standby = False
        self.stage = 'game_over_area'
        self.sub_title.config(text='Game Over Area')
        self.sub_text.config(text='Select with your mouse score detection area\n'
                                  'click q to close wizard')
        self.sub_text.update()
        self.sub_title.update()

    def set_bird_area(self):
        """
        Sets up the area for detecting birds.
        """
        self.standby = False
        self.stage = 'bird_area'
        self.sub_title.config(text='Bird Area')
        self.sub_text.config(text='Select with your mouse score detection area\n'
                                  'click q to close wizard')
        self.sub_text.update()
        self.sub_title.update()

    def set_daytime_area(self):
        """
        Sets up the area for detecting daytime.
        """
        self.standby = False
        self.stage = 'daytime_area'
        self.sub_title.config(text='Daytime Area')
        self.sub_text.config(text='Select with your mouse score detection area\n'
                                  'click q to close wizard')
        self.sub_text.update()
        self.sub_title.update()

    def set_score_area(self):
        """
        Sets up the area for detecting the score.
        """
        self.standby = False
        self.stage = 'score_area'
        self.sub_title.config(text='Score Area')
        self.sub_text.config(text='Select with your mouse score detection area\n'
                                  'click q to close wizard')
        self.sub_text.update()
        self.sub_title.update()

    def mouse_callback(self, event, x, y, _t1, _t2):
        """
        Handles mouse events for selecting areas.
        """
        if event == cv2.EVENT_LBUTTONDOWN and not self.drawing:
            self.drawing = True
            self.selection_start = (x, y)
            self.selection_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.selection_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.standby = True
            self.sub_text.configure(text=f'{self.get_region()}\n'
                                         f'color-mean:{self.trex.test_area(self.get_region(), self.screen)}\n'
                                         f'Press enter to save\n'
                                         f'Use mouse to reselect\n'
                                         f'click q to close wizard', font=('Arial', 12))
            self.sub_text.update()

    def get_region(self):
        """
        Returns the selected region as a dictionary.
        """
        return {'left': min(self.selection_start[0], self.selection_end[0]),
                'top': min(self.selection_start[1], self.selection_end[1]),
                'width': max(self.selection_start[0], self.selection_end[0]) - min(self.selection_start[0],
                                                                                   self.selection_end[0]),
                'height': max(self.selection_start[1], self.selection_end[1]) - min(self.selection_start[1],
                                                                                    self.selection_end[1])}

    def show_area_preview(self):
        """
        Shows a preview of the selected areas.
        """

        def show_preview():
            self.sub_window.iconify()
            self.window.iconify()
            time.sleep(0.2)
            screen = np.array(self.cap.grab(self.cap.monitors[0]))
            # Draw rectangles for each area
            cv2.rectangle(screen, (self.trex.obstacle_area['left'], self.trex.obstacle_area['top']), (
                self.trex.obstacle_area['left'] + self.trex.obstacle_area['width'],
                self.trex.obstacle_area['height'] + self.trex.obstacle_area['top']), (255, 0, 0), 1)
            cv2.rectangle(screen, (self.trex.clear_area['left'], self.trex.clear_area['top']), (
                self.trex.clear_area['left'] + self.trex.clear_area['width'],
                self.trex.clear_area['height'] + self.trex.clear_area['top']), (0, 255, 0), 2)
            cv2.rectangle(screen, (self.trex.game_over_area['left'], self.trex.game_over_area['top']), (
                self.trex.game_over_area['left'] + self.trex.game_over_area['width'],
                self.trex.game_over_area['height'] + self.trex.game_over_area['top']), (0, 0, 255), 3)
            cv2.rectangle(screen, (self.trex.bird_area['left'], self.trex.bird_area['top']), (
                self.trex.bird_area['left'] + self.trex.bird_area['width'],
                self.trex.bird_area['height'] + self.trex.bird_area['top']), (255, 0, 255), 3)
            cv2.rectangle(screen, (self.trex.daytime_area['left'], self.trex.daytime_area['top']), (
                self.trex.daytime_area['left'] + self.trex.daytime_area['width'],
                self.trex.daytime_area['height'] + self.trex.daytime_area['top']), (255, 0, 255), 3)
            cv2.rectangle(screen, (self.trex.score_area['left'], self.trex.score_area['top']), (
                self.trex.score_area['left'] + self.trex.score_area['width'],
                self.trex.score_area['height'] + self.trex.score_area['top']), (255, 0, 255), 3)

            while True:
                cv2.imshow('test', screen)
                key = cv2.waitKey(1)
                if key == 13 or key == ord('q'):
                    cv2.destroyAllWindows()
                    self.sub_window.destroy()
                    self.window.deiconify()
                    break

        time.sleep(0.1)

        self.cap = mss.mss()

        self.sub_window = tk.Tk()
        self.sub_window.focus_force()

        self.sub_title = tk.Label(self.sub_window, text='Press enter to capture game screen\n'
                                                        '\nPress q to close the preview', font=('Arial', 17),
                                  padx=20, pady=20)
        self.sub_title.pack()

        self.sub_text = tk.Label(self.sub_window, text='', font=('Arial', 13), padx=20, pady=30)
        self.sub_text.pack()

        self.sub_window.bind('<Return>', lambda _: show_preview())
        self.sub_window.bind('<q>', lambda _: self.sub_window.destroy())

    def play(self):
        """
        Starts or stops the game based on the current state.
        """
        if self.trex.check_status():
            self.play_button.config(text='stop play', command=self.stop)
            self.play_button.update()
            self.trex.run(self.window)
        else:
            self.message.config(text='Environment is not set', foreground='#ff0000')
            self.message.place(x=110, y=50)

    def stop(self):
        """
        Stops the auto bot and updates the play button text.
        """
        self.trex.stop()
        self.play_button.config(text='start play', command=self.play)
        self.play_button.update()


if __name__ == '__main__':
    gui = GUI()
