#  # # <!-- // /*  SPDX-License-Identifier: blessing */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# import array
import sounddevice as sd

from textual import work
from textual.reactive import reactive

from textual_plotext import PlotextPlot


class VoicePanel(PlotextPlot):  # (PlotWidget)
    """Create an unselectable display element"""

    ALLOW_SELECT = False
    audio = [0]
    sample_freq: int = 16000
    duration: float = 3.0
    sample_len: reactive[float] = reactive(0.0, recompose=True)

    def on_mount(self):
        self.can_focus = True

    @work(exclusive=True)
    async def record_audio(self) -> None:
        """Get audio from mic"""
        precision = self.duration * self.sample_freq
        self.audio = [0]
        self.audio = sd.rec(int(precision), samplerate=self.sample_freq, channels=1)
        sd.wait()
        self.graph_audio()
        # self.calculate_sample_length()

    @work(exclusive=True)
    async def graph_audio(self):
        """Draw audio waveform"""
        self.plt.clear_data()
        self.plt.frame(0)
        self.plt.canvas_color((0, 0, 0))
        self.can_focus = True
        self.plt.xfrequency("0", "0")
        self.plt.yfrequency("0", "0")
        self.plt.scatter(self.audio[:, 0], marker="braille", color=(128, 0, 255))

    @work(exclusive=True)
    async def play_audio(self):
        """Playback audio recordings"""
        try:
            sd.play(self.audio, samplerate=self.sample_freq)
            sd.wait()
        except TypeError as error_log: # todo: map error logger
            print(error_log)

    @work(exclusive=True)
    async def erase_audio(self):
        """Clear audio graph and recording"""
        self.plt.clear_data()
        self.audio = [0]

    # @work(exclusive=True)
    # async def calculate_sample_length(self):
    #     """"""
    #     sample_len = float(len(self.audio) / self.sample_freq)
    #     self.sample_len = sample_len
    #     self.refresh()
