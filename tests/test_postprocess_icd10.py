from services.coder_api.postprocess_icd10 import postprocess_candidates


def test_m545_maps_unspecified():
    cands = [{"code": "M54.5", "title": "Low back pain", "score": 0.89}]
    out = postprocess_candidates(cands, "Low back pain, denies pneumonia.")
    assert out[0]["code"] == "M54.50"


def test_m545_maps_vertebrogenic():
    cands = [{"code": "M54.5", "title": "Low back pain", "score": 0.89}]
    out = postprocess_candidates(cands, "Patient with vertebrogenic low back pain.")
    assert out[0]["code"] == "M54.51"
