import re
from sentence_transformers import SentenceTransformer, util

class Toolset:
    def __init__(self, glossary_path="prompts/domain_glossary.csv"):
        self.glossary = self._load_glossary(glossary_path)
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

    def _load_glossary(self, path):
        glossary = {}
        with open(path, 'r', encoding='utf-8') as f:
            # Skip header line
            lines = f.read().strip().split("\n")[1:]
            for line in lines:
                if not line.strip():
                    continue
                english, filipino, *notes = line.split(",")
                glossary[english.strip()] = filipino.strip()
        return glossary

    def check_domain_terms(self, source, translation):
        missing_terms = []
        incorrect_terms = []
        for eng_term, fil_terms in self.glossary.items():
            fil_variants = [term.strip() for term in fil_terms.split('|')]
            if re.search(rf"\b{re.escape(eng_term)}\b", source, flags=re.IGNORECASE):
                if not any(variant.lower() in translation.lower() for variant in fil_variants):
                    missing_terms.append({"english": eng_term, "expected_filipino": fil_terms})
                elif re.search(rf"\b{re.escape(eng_term)}\b", translation, flags=re.IGNORECASE):
                    incorrect_terms.append({"english": eng_term, "found_in_translation": eng_term})
        return {"missing_terms": missing_terms, "incorrect_terms": incorrect_terms}

    def semantic_similarity(self, source, translation):
        """Calculate semantic similarity between English source and Filipino translation."""
        embedding_src = self.embedding_model.encode(source, convert_to_tensor=True)
        embedding_trans = self.embedding_model.encode(translation, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(embedding_src, embedding_trans).item()
        return {"semantic_similarity_score": round(similarity_score, 4)}

    def alignment_check(self, source, translation):
        """Use semantic similarity as the alignment score instead of length ratio."""
        sim_result = self.semantic_similarity(source, translation)
        score = sim_result["semantic_similarity_score"]
        warnings = []
        if score < 0.7:
            warnings.append("Low semantic similarity; translation may be incomplete or inaccurate.")
        return {"alignment_score": score, "warnings": warnings}

    def grammar_check(self, source):
        issues = []
        text = source

        if not text or not text.strip():
            return {"grammar_flag": True, "notes": ["Empty text"]}

        text = text.strip()

        # 1. Check if text starts with uppercase letter
        if not text[0].isupper():
            issues.append("Text should start with an uppercase letter")

        # 2. Check if text ends with proper punctuation (., !, ?)
        if not re.search(r'[.!?]$', text):
            issues.append("Text should end with punctuation (., !, or ?)")

        # 3. Check each sentence capitalization
        sentences = re.split(r'(?<=[.!?]) +', text)
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if sent and not sent[0].isupper():
                issues.append(f"Sentence {i+1} should start with an uppercase letter")

        # Helper to check preceding word ending rule
        def check_preceding_word(token, should_end_vowel=True):
            matches = re.finditer(rf'\b{token}\b', text, flags=re.IGNORECASE)
            for m in matches:
                start_idx = m.start()
                before_text = text[:start_idx].rstrip()
                prev_word_match = re.search(r'(\b\w+)$', before_text)
                if prev_word_match:
                    prev_word = prev_word_match.group(1)
                    ends_with_vowel = bool(re.search(r'[aeiouyAEIOUY]$', prev_word))
                    if should_end_vowel and not ends_with_vowel:
                        issues.append(f"'{token}' should be preceded by a word ending in a vowel (found '{prev_word}')")
                    elif not should_end_vowel and ends_with_vowel:
                        issues.append(f"'{token}' should be preceded by a word ending in a consonant (found '{prev_word}')")

        # 5. Check 'raw' and 'rin' (preceded by word ending with vowel)
        for token in ['raw', 'rin']:
            check_preceding_word(token, should_end_vowel=True)

        # 6. Check 'daw' and 'din' (preceded by word ending with consonant)
        for token in ['daw', 'din']:
            check_preceding_word(token, should_end_vowel=False)

        return {"grammar_flag": bool(issues), "notes": issues}

    def cultural_check(self, source, translation):
        warnings = []
        # Only apply politeness check to sentences with requests, greetings, or formal context
        polite_context_keywords = ['please', 'thank you', 'sir', 'madam', 'request', 'greetings']
        if any(word in source.lower() for word in polite_context_keywords):
            if "po" not in translation.lower() and "opo" not in translation.lower():
                warnings.append("Polite form markers ('po', 'opo') missing in context that requires respect")
        return {"warnings": warnings}

