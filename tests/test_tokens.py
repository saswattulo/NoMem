from nomem.tokens import ApproxCounter, TokenCounter


def test_approx_counter_is_deterministic():
    c = ApproxCounter()
    assert c.count("hello world") == c.count("hello world")
    assert c.count("") == 0
    # ~4 chars/token
    assert c.count("a" * 40) == 10


def test_count_message_adds_overhead():
    c = ApproxCounter()
    raw = c.count("user: hi")
    assert c.count_message("user", "hi") == raw + c.per_message_overhead


def test_custom_counter_subclass():
    class CharCounter(TokenCounter):
        per_message_overhead = 0

        def count(self, text: str) -> int:
            return len(text)

    c = CharCounter()
    assert c.count("abcd") == 4
    assert c.count_message("user", "hi") == len("user: hi")
