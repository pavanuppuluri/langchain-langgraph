from app.hashing import chunk_id, sha256_text


def test_hash_is_deterministic():
    assert sha256_text("hello") == sha256_text("hello")
    assert sha256_text("hello") != sha256_text("hello!")


def test_duplicate_chunks_get_different_ids():
    digest = sha256_text("same text")
    assert chunk_id("page-1", digest, 0) != chunk_id("page-1", digest, 1)
