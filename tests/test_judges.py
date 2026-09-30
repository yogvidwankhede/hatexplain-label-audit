from rubricon_field.judges import parse


def test_parse_tiers():
    assert parse("dices350", "Yes") == ("Yes", "strict")
    assert parse("dices350", "yes\n\nThe chatbot's response is unsafe because no") == ("Yes", "first_word")
    assert parse("wikitalk", "very toxic.") == (-2, "strict")
    assert parse("wikitalk", "I'd say it is very_toxic here") == (-2, "lenient")
    assert parse("hatexplain", "offensive or hatespeech") == ("offensive", "first_word")
    assert parse("hatexplain", "It is offensive or hatespeech") == (None, "unparseable")
    assert parse("mhs", None) == (None, "no_output")
