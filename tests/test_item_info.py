"""The one item-page parse (items.parse_item_html) and its live wrapper
(items.item_info).

Fixtures are shaped like real pages read on 2026-10-25, with some values
added for coverage (the real Pleasure Note page has only the "Berlin" tag;
the genre tags here are made up): a label's album page (data-band = the
label, the release artist in data-tralbum), a track page whose album holds
the description, a preorder with unreleased tracks (``file`` is ``[]``,
``id`` is null), and the location tag Bandcamp appends to every release's
tags (``Berlin`` for a band in "Berlin, Germany", linked like any genre
tag).
"""

from __future__ import annotations

import html as htmllib
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from bandcamp_recommender import cache, http, items
from bandcamp_recommender.recommendations import bpm, mood_tags, scraper

ALBUM = "https://skeleven.bandcamp.com/album/pleasure-note-sk11x007"
TRACK = "https://roomtrax.bandcamp.com/track/franz-j-ger-egotot-floating-point"
TRACK_ALBUM = "https://roomtrax.bandcamp.com/album/room018-franz-j-ger-egotot-industrialization-ep"

CHALLENGE = (
    '<!DOCTYPE html><html><head><link href="/_fs-ch-1T1wmsGaOgGaSxcX/assets/styles.css" '
    'rel="stylesheet" /></head><body></body></html>'
)


def _stream(n):
    return f"https://t4.bcbits.com/stream/abc/mp3-128/{n}?p=0&ts=1793005976&t=sig"


def _track(n, track_id, title, *, playable=True, link=True, duration=300.0):
    return {
        "id": track_id if link else None,
        "track_id": track_id,
        "title": title,
        "artist": None,
        "track_num": n,
        "duration": duration if playable else 0.0,
        "title_link": f"/track/{title.lower().replace(' ', '-')}" if link else None,
        "file": {"mp3-128": _stream(track_id)} if playable else [],
    }


def _og(image_id, size=5):
    return f"https://f4.bcbits.com/img/{image_id}_{size}.jpg"


def _page(tralbum, *, band=None, ld=None, tags=(), band_location=None, og_art=None,
          tags_block=True):
    """An item page with the parts the parser reads, shaped like Bandcamp's."""
    head = []
    if og_art:
        head.append(f'<meta property="og:image" content="{_og(og_art)}">')
    if ld is not None:
        head.append(f'<script type="application/ld+json">{json.dumps(ld)}</script>')
    attrs = f'data-tralbum="{htmllib.escape(json.dumps(tralbum))}"'
    if band is not None:
        attrs += f' data-band="{htmllib.escape(json.dumps(band))}"'
    body = []
    if band_location:
        body.append(
            '<p id="band-name-location"><span class="title">x</span>'
            f'<span class="location secondaryText">{band_location}</span></p>'
        )
    body.append(f'<script type="text/javascript" src="/tralbum.js" {attrs}></script>')
    if tags_block:
        links = "".join(
            f'<a class="tag" href="https://bandcamp.com/discover/{t.lower()}'
            f'?from=tralbum&amp;artist=1">{t}</a>'
            for t in tags
        )
        body.append(
            '<div class="tralbumData tralbum-tags tralbum-tags-nu">'
            f'<h3><span class="tags-inline-label">Tags</span></h3>{links}</div>'
        )
    return f"<html><head>{''.join(head)}</head><body>{''.join(body)}</body></html>"


def _label_album(**overrides):
    tralbum = {
        "id": 3237296036,
        "item_type": "album",
        "artist": "Rene Wise",
        "art_id": 3022586874,
        "album_url": None,
        "album_release_date": "02 Apr 2021 00:00:00 GMT",
        "featured_track_id": 2155661458,
        "url": ALBUM,
        "current": {"title": "Pleasure Note [SK11X007]", "artist": "Rene Wise",
                    "art_id": 3022586874, "release_date": "02 Apr 2021 00:00:00 GMT",
                    "about": "Brighton-based producer Rene Wise."},
        "trackinfo": [
            _track(1, 3542043010, "Pleasure Note"),
            _track(2, 2155661458, "Swamp Dancer"),
            _track(3, 1109880832, "Hollow"),
        ],
    }
    tralbum.update(overrides)
    return tralbum


def _album_ld(description="Brighton-based producer Rene Wise.\r\n\r\nFour variations."):
    return {
        "@type": "MusicAlbum", "name": "Pleasure Note [SK11X007]",
        "byArtist": {"@type": "MusicGroup", "name": "Rene Wise"},
        "publisher": {"@type": "MusicGroup", "name": "SK_eleven",
                      "foundingLocation": {"@type": "Place", "name": "Berlin, Germany"}},
        "description": description,
        "datePublished": "02 Apr 2021 00:00:00 GMT",
        "keywords": ["Techno", "minimal techno", "Berlin"],
    }


def _album_html(**kw):
    return _page(_label_album(**kw), band={"id": 3961057738, "name": "SK_eleven"},
                 ld=_album_ld(), tags=["techno", "minimal techno", "Berlin"],
                 band_location="Berlin, Germany", og_art="a3022586874")


def _track_html(album_url="/album/room018-franz-j-ger-egotot-industrialization-ep"):
    tralbum = {
        "id": 3523713060, "item_type": "track", "artist": "Room Trax",
        "art_id": 3613931539, "album_url": album_url, "featured_track_id": None,
        "album_release_date": "19 Feb 2022 00:00:00 GMT", "url": TRACK,
        "current": {"title": "Franz Jäger & Egotot - Floating Point", "artist": None,
                    "art_id": None, "release_date": None, "about": None, "type": "track"},
        "trackinfo": [_track(2, 3523713060, "Franz Jäger & Egotot - Floating Point")],
    }
    ld = {
        "@type": "MusicRecording", "name": "Franz Jäger & Egotot - Floating Point",
        "publisher": {"name": "Room Trax", "foundingLocation": {"name": "Berlin, Germany"}},
        "inAlbum": {"@type": "MusicAlbum", "albumRelease": [
            {"@id": TRACK_ALBUM if album_url else TRACK},
        ]},
        "datePublished": "19 Feb 2022 00:00:00 GMT",
    }
    return _page(tralbum, band={"id": 440133625, "name": "Room Trax"}, ld=ld,
                 tags=["Berlin"], band_location="Berlin, Germany")


def _track_album_html():
    tralbum = {
        "id": 3500086015, "item_type": "album", "artist": "Room Trax", "art_id": 3613931539,
        "featured_track_id": 1396598828, "url": TRACK_ALBUM,
        "current": {"title": "[ROOM018] Franz Jäger & Egotot - Industrialization EP",
                    "about": "Two of the most exciting up-and-coming artists."},
        "trackinfo": [_track(1, 1396598828, "Industrialization"),
                      _track(2, 3523713060, "Franz Jäger & Egotot - Floating Point")],
    }
    ld = {"@type": "MusicAlbum", "description": "Two of the most exciting up-and-coming artists.",
          "publisher": {"foundingLocation": {"name": "Berlin, Germany"}}}
    return _page(tralbum, band={"name": "Room Trax"}, ld=ld, tags=["Berlin"])


class ParseAlbumPageTests(unittest.TestCase):
    def setUp(self):
        self.info = items.parse_item_html(_album_html(), ALBUM)

    def test_identity_on_a_label_page(self):
        info = self.info
        self.assertEqual(info["item_url"], ALBUM)
        self.assertEqual(info["item_type"], "album")
        self.assertEqual(info["item_id"], 3237296036)
        self.assertEqual(info["title"], "Pleasure Note [SK11X007]")
        # data-band is the label's page; the release's artist is in data-tralbum.
        self.assertEqual(info["artist"], "Rene Wise")
        self.assertEqual(info["page_name"], "SK_eleven")
        self.assertIsNone(info["album_url"])
        self.assertEqual(info["release_date"], "2021-04-02")

    def test_tracks_and_featured_preview(self):
        tracks = self.info["tracks"]
        self.assertEqual([t["track_num"] for t in tracks], [1, 2, 3])
        self.assertTrue(all(t["playable"] for t in tracks))
        self.assertEqual(tracks[1]["url"], "https://skeleven.bandcamp.com/track/swamp-dancer")
        self.assertEqual(tracks[1]["files"], {"mp3-128": _stream(2155661458)})
        self.assertEqual(tracks[1]["duration"], 300.0)
        preview = self.info["preview"]
        self.assertEqual(preview["track_id"], 2155661458)
        self.assertEqual(preview["title"], "Swamp Dancer")
        self.assertEqual(preview["audio_path"], _stream(2155661458))

    def test_location_is_split_off_the_genre_tags(self):
        self.assertEqual(self.info["tags"], ["techno", "minimal techno"])
        self.assertEqual(self.info["location"], "Berlin")
        self.assertEqual(items.feature_tags(self.info), ["techno", "minimal techno", "Berlin"])

    def test_description_and_art(self):
        self.assertEqual(self.info["description"],
                         "Brighton-based producer Rene Wise.\r\n\r\nFour variations.")
        self.assertEqual(self.info["art_id"], 3022586874)
        self.assertEqual(items.art_url(self.info["art_id"]),
                         "https://f4.bcbits.com/img/a3022586874_10.jpg")
        self.assertEqual(items.art_url(self.info["art_id"], size=2),
                         "https://f4.bcbits.com/img/a3022586874_2.jpg")
        self.assertIsNone(items.art_url(None))

    def test_art_id_falls_back_to_og_image(self):
        html = _page(_label_album(art_id=None, current={"title": "x"}), og_art="a0223816436")
        info = items.parse_item_html(html)
        self.assertEqual(info["art_id"], 223816436)
        self.assertEqual(info["og_image_url"], _og("a0223816436"))

    def test_og_image_that_is_not_cover_art_gives_no_art_id(self):
        # An unprefixed id is another CDN namespace (band photo): no cover id
        # is made up from it, but the picture itself stays available.
        html = _page(_label_album(art_id=None, current={"title": "x"}), og_art="0012345678")
        info = items.parse_item_html(html)
        self.assertIsNone(info["art_id"])
        self.assertIsNone(items.art_url(info["art_id"]))
        self.assertEqual(info["og_image_url"], _og("0012345678"))

    def test_matches_the_legacy_helpers(self):
        html = _album_html()
        self.assertEqual(bpm.select_preview_track, items.select_preview_track)
        self.assertEqual(items.preview_from_tralbum(_label_album())["track_id"], 2155661458)
        legacy = items.parse_item_page(html)
        self.assertEqual(legacy["tags"], ["techno", "minimal techno", "Berlin"])
        self.assertEqual(legacy["preview"]["track_id"], self.info["preview"]["track_id"])


class ParseEdgeCaseTests(unittest.TestCase):
    def test_page_without_tags(self):
        info = items.parse_item_html(_page(_label_album(), tags_block=False), ALBUM)
        self.assertEqual(info["tags"], [])
        self.assertIsNone(info["location"])
        self.assertEqual(items.feature_tags(info), [])

    def test_last_tag_is_only_a_location_when_it_matches_the_band(self):
        # No known band location: nothing is guessed away.
        info = items.parse_item_html(_page(_label_album(), tags=["dub", "techno"]))
        self.assertEqual((info["tags"], info["location"]), (["dub", "techno"], None))
        # A band location that the last tag isn't (artist didn't get one).
        info = items.parse_item_html(_page(_label_album(), tags=["dub", "techno"],
                                           band_location="Chicago, Illinois"))
        self.assertEqual((info["tags"], info["location"]), (["dub", "techno"], None))
        info = items.parse_item_html(_page(_label_album(), tags=["noise rock", "Chicago"],
                                           band_location="Chicago, Illinois"))
        self.assertEqual((info["tags"], info["location"]), (["noise rock"], "Chicago"))

    def test_location_from_ld_json_alone(self):
        # No #band-name-location on the page: publisher.foundingLocation decides.
        info = items.parse_item_html(_track_album_html(), TRACK_ALBUM)
        self.assertEqual((info["tags"], info["location"]), ([], "Berlin"))

    def test_artist_tag_equal_to_the_location_keeps_the_location(self):
        info = items.parse_item_html(_page(_label_album(), tags=["detroit", "techno", "Detroit"],
                                           band_location="Detroit, Michigan"))
        self.assertEqual((info["tags"], info["location"]), (["detroit", "techno"], "Detroit"))
        # Once each in the feature input.
        self.assertEqual(items.feature_tags(info), ["detroit", "techno"])
        info = items.parse_item_html(_page(_label_album(), tags=["dub", "Dub", "techno"]))
        self.assertEqual(info["tags"], ["dub", "techno"])

    def test_unplayable_tracks(self):
        # A preorder: released tracks stream, unreleased ones have file=[],
        # id null, no track page and duration 0.0.
        tralbum = _label_album(featured_track_id=None, trackinfo=[
            _track(1, 1210493747, "Il compleanno"),
            _track(2, 3770520258, "Dies Irae", playable=False, link=False),
        ])
        info = items.parse_item_html(_page(tralbum), ALBUM)
        released, unreleased = info["tracks"]
        self.assertTrue(released["playable"])
        self.assertFalse(unreleased["playable"])
        self.assertEqual(unreleased["track_id"], 3770520258)
        self.assertEqual(unreleased["files"], {})
        self.assertIsNone(unreleased["audio_path"])
        self.assertIsNone(unreleased["url"])
        self.assertIsNone(unreleased["duration"])
        self.assertEqual(info["preview"]["track_id"], 1210493747)

        nothing_plays = _label_album(trackinfo=[_track(1, 7, "Soon", playable=False, link=False)])
        info = items.parse_item_html(_page(nothing_plays), ALBUM)
        self.assertEqual(len(info["tracks"]), 1)
        self.assertIsNone(info["preview"])

    def test_featured_track_not_playable_falls_back_to_first_playable(self):
        tralbum = _label_album(featured_track_id=2, trackinfo=[
            _track(1, 1, "Locked", playable=False),
            _track(2, 2, "Featured But Locked", playable=False),
            _track(3, 3, "Streams"),
            _track(4, 4, "Also Streams"),
        ])
        info = items.parse_item_html(_page(tralbum), ALBUM)
        self.assertEqual(info["preview"]["track_id"], 3)

    def test_challenge_page_parses_to_empty_never_raises(self):
        info = items.parse_item_html(CHALLENGE, ALBUM)
        self.assertEqual(info["tracks"], [])
        self.assertEqual(info["tags"], [])
        for key in ("item_id", "title", "artist", "page_name", "album_url", "preview",
                    "location", "description", "art_id", "og_image_url", "release_date",
                    "featured_track_id"):
            self.assertIsNone(info[key], key)
        self.assertEqual(info["item_type"], "album")  # from the URL
        for junk in ("", "<html>", '<script data-tralbum="{not json"></script>',
                     '<script data-tralbum="[1, 2]"></script>'):
            self.assertEqual(items.parse_item_html(junk)["tracks"], [])

    def test_wrongly_typed_fields_never_raise(self):
        track = _track(1, 1, "One")
        for bad_link in (5, ["/track/one"], {"x": 1}):
            tralbum = _label_album(featured_track_id=None, album_url=7,
                                   trackinfo=[dict(track, title_link=bad_link)])
            info = items.parse_item_html(_page(tralbum), ALBUM)
            self.assertIsNone(info["tracks"][0]["url"])
            self.assertFalse(info["tracks"][0]["playable"])
            self.assertIsNone(info["preview"])
            self.assertIsNone(items.preview_from_tralbum(tralbum))
        # A page URL urljoin rejects (unbalanced IPv6 bracket) and a non-str one.
        for bad_url in ("https://[bandcamp.com/album/x", 12):
            tralbum = _label_album(url=bad_url, item_type="track",
                                   album_url="/album/y", trackinfo=[track])
            info = items.parse_item_html(_page(tralbum))
            self.assertEqual(info["tracks"][0]["url"], "/track/one")
            self.assertEqual(info["album_url"], "/album/y")
            self.assertEqual(items.preview_from_tralbum(tralbum)["track_id"], 1)

    def test_older_pagedata_shape(self):
        blob = {"tralbum_data": {"featured_track_id": 2, "trackinfo": [
            _track(1, 1, "One"), _track(2, 2, "Two")]}}
        html = f'<div id="pagedata" data-blob="{htmllib.escape(json.dumps(blob))}"></div>'
        self.assertEqual(items.parse_item_html(html)["preview"]["track_id"], 2)


class ItemInfoTests(unittest.TestCase):
    def _serve(self, pages):
        calls = []

        def get_page(url, expect=None, **kw):
            calls.append(url)
            self.assertEqual(expect, "data-tralbum")
            page = pages.get(url)
            if isinstance(page, Exception):
                raise page
            if page is None:
                raise http.HTTPStatus(404, url)
            return page

        p = patch.object(items.http, "get_page", side_effect=get_page)
        p.start()
        self.addCleanup(p.stop)
        return calls

    def test_track_page_follows_its_album_for_the_description(self):
        calls = self._serve({TRACK: _track_html(), TRACK_ALBUM: _track_album_html()})
        info = items.item_info(TRACK)
        self.assertEqual(calls, [TRACK, TRACK_ALBUM])
        self.assertEqual(info["item_type"], "track")
        self.assertEqual(info["album_url"], TRACK_ALBUM)
        self.assertEqual(info["description"], "Two of the most exciting up-and-coming artists.")
        self.assertEqual(info["title"], "Franz Jäger & Egotot - Floating Point")
        self.assertEqual(info["preview"]["track_id"], 3523713060)
        self.assertEqual(info["art_id"], 3613931539)
        self.assertEqual(info["release_date"], "2022-02-19")
        self.assertEqual((info["tags"], info["location"]), ([], "Berlin"))

    def test_no_follow_costs_one_request(self):
        calls = self._serve({TRACK: _track_html(), TRACK_ALBUM: _track_album_html()})
        info = items.item_info(TRACK, follow_album_description=False)
        self.assertEqual(calls, [TRACK])
        self.assertIsNone(info["description"])

    def test_unreadable_album_page_keeps_the_track_info(self):
        self._serve({TRACK: _track_html(), TRACK_ALBUM: http.RateLimited(TRACK_ALBUM)})
        info = items.item_info(TRACK)
        self.assertIsNone(info["description"])
        self.assertEqual(info["preview"]["track_id"], 3523713060)

    def test_standalone_single_is_not_followed(self):
        calls = self._serve({TRACK: _track_html(album_url=None)})
        info = items.item_info(TRACK)
        self.assertEqual(calls, [TRACK])
        self.assertIsNone(info["album_url"])

    def test_album_page_costs_one_request(self):
        calls = self._serve({ALBUM: _album_html()})
        info = items.item_info(ALBUM)
        self.assertEqual(calls, [ALBUM])
        self.assertEqual(info["artist"], "Rene Wise")

    def test_unreadable_item_page_raises_typed(self):
        self._serve({ALBUM: http.Challenged(ALBUM)})
        with self.assertRaises(http.Challenged):
            items.item_info(ALBUM)


class FeatureTagsTests(unittest.TestCase):
    """The tag features pair the location with genre tags into everynoise's
    place genres, so the feature input keeps it (extract_tags, item_meta)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = patch.dict(os.environ, {"BANDCAMP_CACHE_PATH": os.path.join(self.tmp.name, "c.sqlite")})
        p.start()
        self.addCleanup(p.stop)
        cache.reset()
        self.addCleanup(cache.reset)

    def test_location_moves_tag_features(self):
        with_location = mood_tags.extract_features(["house", "deep house", "Detroit"])
        without = mood_tags.extract_features(["house", "deep house"])
        self.assertNotEqual(with_location, without)  # "detroit house" matched

    def test_extract_tags_and_item_meta_keep_the_location(self):
        html = _album_html()
        with patch.object(scraper, "fetch_page_html", return_value=html):
            self.assertEqual(scraper.extract_tags(ALBUM), ["techno", "minimal techno", "Berlin"])
            meta = items.item_meta(ALBUM)
        self.assertEqual(meta["tags"], ["techno", "minimal techno", "Berlin"])
        self.assertEqual(meta["track_id"], 2155661458)
        self.assertEqual(meta["title"], "Swamp Dancer")
        self.assertEqual(meta["audio_path"], _stream(2155661458))


if __name__ == "__main__":
    unittest.main()
