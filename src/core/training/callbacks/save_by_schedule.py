from typing import override

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class SaveByScheduleCallback(TrainerCallback):
    def __init__(self, schedule: list[int]):
        self.schedule = schedule

    @override
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        assert state.epoch is not None

        epoch_num = int(state.epoch)

        control.should_save = False
        if epoch_num in self.schedule:
            control.should_save = True
