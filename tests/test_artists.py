"""Artist identity (bandcamp_recommender.artists.artist_tokens).

Ported from boemketel_radio's tests/test_utils.py, where this function lived
before it moved into the library."""

from bandcamp_recommender.artists import artist_tokens


class TestArtistTokens:
    def test_single_artist(self):
        assert artist_tokens("Siu Mata") == {"siu mata"}
        assert artist_tokens("modest by default | Mabisyo") == {"modest by default", "mabisyo"}

    def test_empty(self):
        assert artist_tokens("") == set()
        assert artist_tokens(None) == set()
        assert artist_tokens("   ") == set()

    def test_collaboration_variants_share_token(self):
        # The whole point: these forms must all yield "siu mata" so artist
        # dedup recognizes them as the same act.
        for s in ("Siu Mata & Amor Satyr", "Amor Satyr & Siu Mata",
                  "Siu Mata, Amor Satyr", "Siu Mata feat. Amor Satyr",
                  "Siu Mata x Amor Satyr", "Siu Mata / Amor Satyr",
                  "Siu Mata + Amor Satyr", "Siu Mata vs. Amor Satyr",
                  "Siu Mata ft Amor Satyr", "Siu Mata | Amor Satyr"):
            assert artist_tokens(s) == {"siu mata", "amor satyr"}, s

    def test_case_insensitive(self):
        assert artist_tokens("SIU MATA") == artist_tokens("Siu Mata")
        assert artist_tokens("Siu Mata FEAT. Amor Satyr") == {"siu mata", "amor satyr"}

    def test_drops_compilation_labels(self):
        for s in ("Various Artists", "VARIOUS ARTISTS", "Various", "V/A", "v/a",
                  "VA", "Unknown Artist", " various artists "):
            assert artist_tokens(s) == set(), s

    def test_compilation_label_inside_a_credit_is_dropped(self):
        assert artist_tokens("Various Artists & Siu Mata") == {"siu mata"}

    def test_does_not_split_on_and_or_with(self):
        # "and"/"with" are common in band names: must stay intact.
        assert artist_tokens("Florence and the Machine") == {"florence and the machine"}
        assert artist_tokens("Dancing with Ghosts") == {"dancing with ghosts"}

    def test_x_only_splits_as_standalone_word(self):
        # "x" inside a word (Aphex) is not a separator.
        assert artist_tokens("Aphex Twin") == {"aphex twin"}
        assert artist_tokens("Xenia Rubinos") == {"xenia rubinos"}
