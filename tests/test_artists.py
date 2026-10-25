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
                  "Siu Mata ft Amor Satyr", "Siu Mata | Amor Satyr",
                  "Siu Mata × Amor Satyr", "Siu Mata×Amor Satyr",
                  "Siu Mata featuring Amor Satyr", "Siu Mata (feat. Amor Satyr)",
                  "Siu Mata (Ft. Amor Satyr)"):
            assert artist_tokens(s) == {"siu mata", "amor satyr"}, s

    def test_case_insensitive(self):
        assert artist_tokens("SIU MATA") == artist_tokens("Siu Mata")
        assert artist_tokens("Siu Mata FEAT. Amor Satyr") == {"siu mata", "amor satyr"}

    def test_drops_compilation_labels(self):
        for s in ("Various Artists", "VARIOUS ARTISTS", "Various", "V/A", "v/a",
                  "VA", "V.A.", "Various Artist", "Unknown Artist", " various artists "):
            assert artist_tokens(s) == set(), s

    def test_compilation_label_inside_a_credit_is_dropped(self):
        assert artist_tokens("Various Artists & Siu Mata") == {"siu mata"}
        assert artist_tokens("V.A. / Ancient Methods") == {"ancient methods"}

    def test_parenthesised_feature_credit_with_several_artists(self):
        assert artist_tokens("Cimm & Youngsta (feat. Riko Dan)") == {"cimm", "youngsta", "riko dan"}
        assert artist_tokens("Ourman (ft. Gnischrew, Off White & SBK)") == {
            "ourman", "gnischrew", "off white", "sbk"}

    def test_other_parentheses_stay_part_of_the_name(self):
        # "(UK)" disambiguates same-named acts: it must not become an artist
        # that merges every "(UK)" act.
        assert artist_tokens("Phase (UK)") == {"phase (uk)"}

    def test_does_not_split_on_and_or_with(self):
        # "and"/"with" are common in band names: must stay intact.
        assert artist_tokens("Florence and the Machine") == {"florence and the machine"}
        assert artist_tokens("Dancing with Ghosts") == {"dancing with ghosts"}

    def test_x_only_splits_as_standalone_word(self):
        # "x" inside a word (Aphex) is not a separator.
        assert artist_tokens("Aphex Twin") == {"aphex twin"}
        assert artist_tokens("Xenia Rubinos") == {"xenia rubinos"}

    def test_word_connectors_only_split_between_spaces(self):
        # A name that starts or ends with "X" keeps it (the radio's copy gave
        # "madam", merging Madam X with any "Madam").
        assert artist_tokens("Madam X") == {"madam x"}
        assert artist_tokens("X-Coast & La Goony Chonga") == {"x-coast", "la goony chonga"}
        assert artist_tokens("Nu Brand Flexxx x Rebound X") == {"nu brand flexxx", "rebound x"}

    def test_separators_inside_a_band_name_split_it(self):
        # Accepted trade-off (see the module docstring): leaning to splitting
        # catches collabs at the cost of rare false merges.
        assert artist_tokens("Earth, Wind & Fire") == {"earth", "wind", "fire"}
        assert artist_tokens("AC/DC") == {"ac", "dc"}
