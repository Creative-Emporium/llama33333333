# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional
import torch

import fire

from llama import Dialog, Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    seed: int = -1,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    #l0_dict = torch.load('./train/20240604-block_0-wo+fn+w1+w2+w3-lr1e-3/model_10-acc5.5827.chkpt', map_location='cpu')
    #generator.model.load_relu_block(0, l0_dict['model'])

    random_gen = torch.Generator(device='cuda')
    if seed >= 0:
        random_gen.manual_seed(seed)

    dialogs: List[Dialog] = [
#        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
#        [
#            {"role": "user", "content": "I am going to Paris, what should I see?"},
#            {
#                "role": "assistant",
#                "content": """\
#Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:
#
#1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
#2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
#3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.
#
#These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
#            },
#            {"role": "user", "content": "What is so great about #1?"},
#        ],
#        [
#            {"role": "system", "content": "Always answer with Haiku"},
#            {"role": "user", "content": "I am going to Paris, what should I see?"},
#        ],
#        [
#            {
#                "role": "system",
#                "content": "Always answer with emojis",
#            },
#            {"role": "user", "content": "How to go from Beijing to NY?"},
#        ],
        [
            {"role": "system", "content": "Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand."},
            {"role": "user", "content": "What is the capital city of France?"},
        ],
    ]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        generator=random_gen,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
