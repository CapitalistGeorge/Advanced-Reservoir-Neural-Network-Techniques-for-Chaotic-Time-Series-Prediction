def test_skccm_public_api_exists():
    import skccm

    # Smoke: пакет импортируется и имеет хотя бы один публичный атрибут
    public = [n for n in dir(skccm) if not n.startswith("_")]
    assert len(public) > 0
