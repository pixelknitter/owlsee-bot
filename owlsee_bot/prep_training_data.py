import pathlib
from datasets import load_dataset
from tqdm import tqdm

# Interesting Data Sets
# https://huggingface.co/datasets/llm-wizard/english_to_pirate
# https://huggingface.co/datasets/Peyton3995/dolly-15k-mistral-pirate
# https://huggingface.co/datasets/marimeireles/scifi-corpus
# https://huggingface.co/datasets/stevez80/Sci-Fi-Books-gutenberg
# https://huggingface.co/datasets/sander-wood/irishman
# https://huggingface.co/datasets/open-llm-leaderboard/details_Monero__WizardLM-Uncensored-SuperCOT-StoryTelling-30b

PROMPT_TEMPLATE = "[INST] <<SYS>>\nWith the given Input and appropriate Context, use the Response to speak like a pirate on any sea (astral, plantary, etc) that they might traverse.\n<</SYS>>\n\nInput:\n{message}\n\nContext: {context} [/INST]\n\nResponse:{response}"


def format_pirate_instruction(sample):
    return {
        "text": PROMPT_TEMPLATE.format(
            message=sample["instruction"],
            context=sample["context"],
            response=sample["response"],
        )
    }


project_root = pathlib.Path(__file__).parent.resolve()

datasets = {
    "pirate": load_dataset("Peyton3995/dolly-15k-mistral-pirate"),
    # "scifi": load_dataset("marimeireles/scifi-corpus"),
    # "storytelling": load_dataset(
    #     "open-llm-leaderboard/details_Monero__WizardLM-Uncensored-SuperCOT-StoryTelling-30b",
    #     "harness_winogrande_5",
    #     split="train",
    # ),
}

for key, data in tqdm(datasets.items()):
    # FIXME: formatting of data is different between pirate, storytelling, and scifi
    if key == "pirate":
        # what's in the data?
        print(format_pirate_instruction(data["train"][1]))
        ds = data.map(
            format_pirate_instruction,
            remove_columns=["category"],
        )
        # ds = {
        #     format_pirate_instruction(row)
        #     for row in tqdm(data, "Formatting piratical instructions...")
        # }
    elif key == "scifi":
        # TODO: handle scifi case
        pass
    elif key == "storytelling":
        # TODO: handle storytelling case
        pass
    else:
        # handle other cases
        pass
    ds.to_json(project_root.__str__() + f"data/{key}.jsonl")
