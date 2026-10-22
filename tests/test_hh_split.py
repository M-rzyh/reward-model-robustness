"""Unit tests for the HH-RLHF transcript splitter (hand-written transcripts, no data needed)."""
import pytest

from src.data.hh import parse_turns, split_transcript
from src.data.schema import valid_turns


def test_single_turn():
    prompt, response = split_transcript("\n\nHuman: hi\n\nAssistant: hello")
    assert prompt == "\n\nHuman: hi"
    assert response == "hello"
    assert parse_turns(prompt) == [{"role": "user", "content": "hi"}]


def test_multi_turn_uses_last_assistant():
    text = "\n\nHuman: a\n\nAssistant: b\n\nHuman: c\n\nAssistant: d"
    prompt, response = split_transcript(text)
    assert response == "d"
    msgs = parse_turns(prompt)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [m["content"] for m in msgs] == ["a", "b", "c"]
    assert valid_turns(msgs)


def test_inline_assistant_word_is_not_a_turn():
    text = "\n\nHuman: my Assistant: app crashed\n\nAssistant: sorry"
    prompt, response = split_transcript(text)
    assert response == "sorry"
    msgs = parse_turns(prompt)
    assert msgs == [{"role": "user", "content": "my Assistant: app crashed"}]


def test_two_assistant_turns_in_a_row_are_invalid():
    msgs = parse_turns("\n\nHuman: a\n\nAssistant: b\n\nAssistant: c\n\nHuman: d")
    assert not valid_turns(msgs)


def test_no_assistant_turn_raises():
    with pytest.raises(ValueError):
        split_transcript("\n\nHuman: nobody answered")