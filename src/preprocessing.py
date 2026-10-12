import re

# Pre-compile regex patterns for emoji and modifier handling
SKIN_TONE = re.compile(r"[\U0001F3FB-\U0001F3FF]")
GENDER    = re.compile(r"[\u2640\u2642]")
FLAG      = re.compile(r"[\U0001F1E6-\U0001F1FF]{2}")
EMOJI     = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]")

def clean_tweet(text: str) -> str:
    """
    Hausa tweet cleaning pipeline:
    - Removes HTML entities, Twitter handles (@user), and URLs.
    - Normalizes African/Hausa glottal apostrophes (’ʼ‘ -> ').
    - Strips skin tones, gender modifiers, and zero-width joiners.
    - Isolates emojis and flags with spaces to preserve sentiment tokens.
    """
    if not isinstance(text, str):
        return ""

    # 1. HTML Entities
    text = re.sub(r"&[a-z]+;", "", text)
    
    # 2. Twitter Handles (@user)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    
    # 3. URLs & Dangling Protocols
    text = re.sub(r"https?://\S+|www\.\S+|https?:(?=\s|$)", "", text)
    
    # 4. Normalize Apostrophes (Crucial for Hausa glottal consonants e.g., ɓ, ɗ, 'y)
    text = re.sub(r"[’ʼ‘]", "'", text)

    # 5. Emoji & Modifier Processing
    text = text.replace("\ufe0f", "").replace("\u200d", "")   # Invisible selectors/joiners
    text = SKIN_TONE.sub("", text)                            # Strip skin tones
    text = GENDER.sub("", text)                               # Strip gender modifiers
    text = FLAG.sub(lambda m: f" {m.group()} ", text)         # Space-out country flags
    text = EMOJI.sub(lambda m: f" {m.group()} ", text)        # Space-out general emojis

    # 6. Normalize Whitespace
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()