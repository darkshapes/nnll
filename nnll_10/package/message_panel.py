# # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
# # <!-- // /*  d a r k s h a p e s */ -->
from textual import work
from textual.widgets import TextArea


class MessagePanel(TextArea):
    # def on_mount(self) -> None:
    #     # self.border_subtitle = "Prompt"
    #     # self.theme = "vscode_dark"
    #     # self.language = "python"

    @work(exclusive=True)
    async def erase_message(self):
        """Empty panel contents"""
        self.clear()
