from backend.pipeline import tts


def test_orpheus_chunking_keeps_adjacent_proper_name_tokens_together():
    source = (
        "In AI research, QbitAI reports that a team from Princeton University, "
        "Ant Group, and Stanford University has a paper."
    )

    chunks = tts._split_tts_text(source, max_words=12)

    assert chunks == [
        "In AI research, QbitAI reports that a team from Princeton University,",
        "Ant Group, and Stanford University has a paper.",
    ]
    assert " ".join(" ".join(chunks).split()) == source
