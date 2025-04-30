#  # # <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# import array
import sounddevice as sd

from textual import work
from textual.binding import Binding
from textual.reactive import reactive

from textual_plotext import PlotextPlot

from nnll_01 import nfo  # , debug_monitor


class VoicePanel(PlotextPlot):  # (PlotWidget)
    """Create an unselectable waveform display element"""

    ALLOW_SELECT = False
    audio = [0]
    sample_freq: int = 16000
    duration: float = 3.0
    sample_len: reactive[float] = reactive(0.0, recompose=True)
    BINDINGS = [
        Binding("alt+bk", "erase_audio()", "del"),
        Binding("enter", "record_audio()", "◉", priority=True),
        Binding("space", "play_audio()", "▶︎", priority=True),
    ]

    def on_mount(self):
        self.can_focus = True
        # self.theme = "flexoki"

    @work(exclusive=True)
    async def record_audio(self) -> None:
        """Get audio from mic"""
        self.plt.clear_data()
        precision = self.duration * self.sample_freq
        self.audio = [0]
        self.audio = sd.rec(int(precision), samplerate=self.sample_freq, channels=1)
        sd.wait()
        self.graph_audio()
        # self.calculate_sample_length()

    @work(exclusive=True)
    async def graph_audio(self):
        """Draw audio waveform"""
        self.plt.frame(0)
        self.plt.canvas_color((0, 0, 0))
        self.can_focus = True
        self.plt.xfrequency("0", "0")
        self.plt.yfrequency("0", "0")
        self.plt.scatter(self.audio[:, 0], marker="braille", color=(128, 0, 255))
        self.time_audio()

    @work(exclusive=True)
    async def play_audio(self):
        """Playback audio recordings"""
        try:
            sd.play(self.audio, samplerate=self.sample_freq)
            sd.wait()
        except TypeError as error_log:
            nfo(error_log)

    @work(exclusive=True)
    async def erase_audio(self):
        """Clear audio graph and recording"""
        self.plt.clear_data()
        self.audio = [0]

    @work(exclusive=True)
    async def time_audio(self):
        sample_len = float(len(self.audio) / self.sample_freq)
        self.sample_len = sample_len
        self.refresh()
