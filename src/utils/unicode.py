from unicodedata import normalize

from ftfy import fix_text, TextFixerConfig


def fix_normalize_unicode(input_str: str, form: str = "NFC") -> str:
    if form == "NFC" or form == "NFKC":
        fixed = fix_text(input_str, TextFixerConfig(normalization=form))
        return fixed
    else:
        fixed = fix_text(input_str)
        norm = normalize(form, fixed)
        return norm


def fix_norm_evidence(evidence: dict, form: str = "NFD") -> dict:
    new_evidence = {k: normalize(form, fix_text(v))
                    for k, v in evidence.items()}
    return new_evidence


def is_norm(str: str, form: str) -> bool:
    if str == normalize(form, str):
        return True
    else:
        return False
