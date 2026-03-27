from deepeyenet.data.dataset import SampleRecord, build_keyword_index
from deepeyenet.data.vocab import Vocabulary
from deepeyenet.training.graph import build_keyword_adjacency


def test_vocab_roundtrip():
    vocab = Vocabulary.build(["hello retina report", "retina finding"])
    ids = vocab.encode("hello retina", max_length=8)
    text = vocab.decode(ids)
    assert "hello" in text
    assert "retina" in text


def test_graph_builder_shape():
    records = [
        SampleRecord("a.jpg", ["a", "b"], "clinical", "report"),
        SampleRecord("b.jpg", ["b", "c"], "clinical", "report"),
    ]
    keyword_to_idx = build_keyword_index(records)
    adjacency = build_keyword_adjacency(records, keyword_to_idx)
    assert adjacency.shape[0] == len(keyword_to_idx)
    assert adjacency.shape[0] == adjacency.shape[1]
