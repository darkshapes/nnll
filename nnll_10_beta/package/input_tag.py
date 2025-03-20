# #  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
# #  # # <!-- // /*  d a r k s h a p e s */ -->

# from textual.reactive import reactive

# from package.carousel import Carousel
# from package.model_registry import VALID_CONVERSIONS


# class InputTag(Carousel):
#     """Input Types"""

#     valid_conversions: dict = VALID_CONVERSIONS
#     # available_inputs: dict = {"Prompt": "message_panel", "Speech": "voice_panel"}
#     current_input: reactive[str] = reactive("")

#     def on_mount(self):
#         self.current_input = next(iter(self.valid_conversions))
#         self.add_columns(("0", "1"))
#         self.add_rows([row.strip()] for row in self.valid_conversions)
#         self.cursor_foreground_priority = "css"
#         self.cursor_background_priority = "css"


    # @work(exclusive=True)
    # async def alternate_panel(self, id_name, y_coordinate):
    #     input_tag = self.query_one("#input_tag")
    #     input_tag.scroll_to(x=0, y=y_coordinate, force=True, immediate=True, on_complete=input_tag.refresh)
    #     self.query_one(PanelSwap).current = id_name

# from package.input_tag import InputTag
        # self.query_one("#voice_panel").focus()

                    # yield InputTag(id="input_tag")
            # current_input: reactive[str] = reactive("")

        # self.current_input = self.query_one("#input_tag").current_input
#   elif self.query_one("#input_tag").has_focus:
#             event.prevent_default()
#             input_tag = self.query_one("#input_tag")
#             class_name = input_tag.emulate_scroll_down(input_tag.available_inputs)
#             self.query_one(PanelSwap).current = class_name
        # elif self.query_one("#input_tag").has_focus:
        #     event.prevent_default()
        #     input_tag = self.query_one("#input_tag")
        #     class_name = input_tag.emulate_scroll_up(input_tag.available_inputs)
        #     self.query_one(PanelSwap).current = class_name