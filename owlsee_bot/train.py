import os
import pathlib
import replicate

user_name = os.getenv("REPLICATE_USERNAME")
project_root = pathlib.Path(__file__).parent.resolve()
# TODO: make this more dynamic
path_to_training = project_root.__str__() + "data/pirate.jsonl"

training = replicate.trainings.create(
    version="meta/meta-llama-3-8b:9a9e68fc8695f5847ce944a5cecf9967fd7c64d0fb8c8af1d5bdcc71f03c5e47",
    input={
        "train_data": path_to_training,
        "num_train_epochs": 3,
    },
    destination=f"{user_name}/llama3-pirate",
)

print(f"Started at {training.started_at}")
print(f"Status: {training.status}")
training.completed_at & print(f"Completed at {training.completed_at}")
