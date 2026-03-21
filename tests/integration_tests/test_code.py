import pytest
import torch

from modern_transformer.transformer.model import Transformer


SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2

INDEX2WORDS = {SOS_TOKEN: "SOS", EOS_TOKEN: "EOS", PAD_TOKEN: "PAD"}

WORDS = "How are you doing ? I am good and you ?"
# Keep vocab deterministic for stable tests.
for word in sorted(set(WORDS.lower().split(" "))):
    INDEX2WORDS[len(INDEX2WORDS)] = word

WORDS2INDEX = {w: i for i, w in INDEX2WORDS.items()}

HIDDEN_SIZE = 12
VOCAB_SIZE = len(WORDS2INDEX)
N_BLOCKS = 10
D_FF = 20
CONTEXT_SIZE = 100
WINDOW_SIZE = 11
NUM_HEADS = 3
NUM_EXPERTS = 10
N_EXPERTS_PER_TOKEN = 2


def convert2tensors(sentence: str, max_len: int) -> torch.Tensor:
    words_list = sentence.lower().split(" ")
    padding = ["PAD"] * (max_len - len(words_list))
    words_list.extend(padding)
    indexes = [WORDS2INDEX[word] for word in words_list]
    return torch.tensor(indexes, dtype=torch.long).view(1, -1)


@pytest.fixture
def transformer() -> Transformer:
    torch.manual_seed(0)
    return Transformer(
        vocabulary_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        window_size=WINDOW_SIZE,
        d_ff=D_FF,
        num_experts=NUM_EXPERTS,
        n_experts_per_token=N_EXPERTS_PER_TOKEN,
        n_blocks=N_BLOCKS,
        max_seq_len=CONTEXT_SIZE,
    )


def test_forward_shape_and_finite_values(transformer: Transformer) -> None:
    input_sentence = "How are you doing ?"
    input_tensor = convert2tensors(input_sentence, CONTEXT_SIZE)

    output = transformer(input_tensor)

    assert output.shape == (1, CONTEXT_SIZE, VOCAB_SIZE)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_predicted_token_is_in_vocab(transformer: Transformer) -> None:
    input_sentence = "How are you doing ?"
    input_tensor = convert2tensors(input_sentence, CONTEXT_SIZE)

    output = transformer(input_tensor)
    # top prediction for each position -> [seq_len, 1]
    _, indexes = output.squeeze(0).topk(1, dim=-1)
    predicted_token = INDEX2WORDS[indexes[5].item()]

    assert predicted_token in WORDS2INDEX
