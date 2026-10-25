"""The one Bandcamp item URL canonicaliser (bandcamp_recommender.items)."""

from bandcamp_recommender.items import normalize_item_url


class TestNormalizeItemUrl:
    def test_variants_collapse_to_one_form(self):
        canonical = "https://x.bandcamp.com/album/y"
        for v in (
            "https://x.bandcamp.com/album/y",
            "https://X.bandcamp.com/album/y/",
            "https://x.bandcamp.com/album/y?from=search",
            "HTTPS://x.bandcamp.com/album/y#top",
            "  https://x.bandcamp.com/album/y/?from=fanpub#x \n",
            "https://x.bandcamp.com/album/Y",
        ):
            assert normalize_item_url(v) == canonical, v

    def test_api_urls_are_unchanged(self):
        # Every URL the fan-collection API returns is already canonical, so
        # the disk-cache keys built from them (item metadata, audio features)
        # are the same as before this function was shared.
        url = "https://music.businesscasual.biz/album/bizcas10-ten-years-of-business-casual"
        assert normalize_item_url(url) == url

    def test_empty_and_bare_strings(self):
        assert normalize_item_url(None) == ""
        assert normalize_item_url("") == ""
        assert normalize_item_url("u0") == "u0"
