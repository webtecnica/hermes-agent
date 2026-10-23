"""Tests for TelegramPlatform._merge_caption caption deduplication logic."""


from plugins.platforms.telegram.adapter import TelegramAdapter

merge = TelegramAdapter._merge_caption


class TestMergeCaptionBasic:
    def test_no_existing_text(self):
        assert merge(None, "Hello") == "Hello"


class TestMergeCaptionSubstringBug:
    """These are the exact scenarios that the old substring check got wrong."""

    def test_shorter_caption_not_dropped_when_substring(self):
        # Bug: "Meeting" in "Meeting agenda" → True → caption was silently lost
        result = merge("Meeting agenda", "Meeting")
        assert result == "Meeting agenda\n\nMeeting"


class TestMergeCaptionWhitespace:
    def test_trailing_space_treated_as_duplicate(self):
        assert merge("Revenue", "Revenue  ") == "Revenue"


class TestMergeCaptionMultipleItems:
    def test_three_unique_captions_all_present(self):
        text = merge(None, "A")
        text = merge(text, "B")
        text = merge(text, "C")
        assert text == "A\n\nB\n\nC"


class TestMergeIntoPendingMediaTextInlined:
    def _event(self, urls, types, inlined=None, text=""):
        from gateway.platforms.event import MessageEvent, MessageType

        ev = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            user_id="u1",
        )
        ev.media_urls = list(urls)
        ev.media_types = list(types)
        if inlined is not None:
            ev.media_text_inlined = list(inlined)
        return ev

    def test_merge_two_documents(self):
        adapter = object.__new__(TelegramAdapter)
        pending = {}
        ev1 = self._event(["doc1.pdf"], ["application/pdf"], inlined=[False])
        ev2 = self._event(["doc2.txt"], ["text/plain"], inlined=[True], text="content")
        adapter._merge_into_pending(pending, "group1", ev1)
        adapter._merge_into_pending(pending, "group1", ev2)

        merged = pending["group1"]
        assert len(merged.media_urls) == 2
        assert len(merged.media_text_inlined) == 2
        assert merged.media_text_inlined == [False, True]

    def test_merge_document_and_photo(self):
        adapter = object.__new__(TelegramAdapter)
        pending = {}
        ev1 = self._event(["doc1.pdf"], ["application/pdf"], inlined=[True])
        ev2 = self._event(["photo1.jpg"], ["image/jpeg"], inlined=None)
        adapter._merge_into_pending(pending, "group1", ev1)
        adapter._merge_into_pending(pending, "group1", ev2)

        merged = pending["group1"]
        assert len(merged.media_urls) == 2
        assert len(merged.media_text_inlined) == 2
        assert merged.media_text_inlined == [True, False]

    def test_merge_photo_and_document(self):
        adapter = object.__new__(TelegramAdapter)
        pending = {}
        ev1 = self._event(["photo1.jpg"], ["image/jpeg"], inlined=[])
        ev2 = self._event(["doc1.txt"], ["text/plain"], inlined=[True])
        adapter._merge_into_pending(pending, "group1", ev1)
        adapter._merge_into_pending(pending, "group1", ev2)

        merged = pending["group1"]
        assert len(merged.media_urls) == 2
        assert len(merged.media_text_inlined) == 2
        assert merged.media_text_inlined == [False, True]

    def test_merge_two_photos_keeps_empty_inlined(self):
        adapter = object.__new__(TelegramAdapter)
        pending = {}
        ev1 = self._event(["photo1.jpg"], ["image/jpeg"], inlined=[])
        ev2 = self._event(["photo2.jpg"], ["image/jpeg"], inlined=[])
        adapter._merge_into_pending(pending, "group1", ev1)
        adapter._merge_into_pending(pending, "group1", ev2)

        merged = pending["group1"]
        assert len(merged.media_urls) == 2
        assert len(merged.media_text_inlined) == 0



