# Copyright (c) ModelScope Contributors. All rights reserved.
# Data-Juicer integration for trajectory quality filtering.
#
# ── Replaces our custom code ───────────────────────────────────────────────────
#   repeat.py  →  word_repeat_filter + char_repeat_filter
#
# ── Complements (our code kept for deeper detection) ──────────────────────────
#   token_soup.py  →  special_chars_filter / alphanumeric_filter (shallower)
#   perplexity.py  →  kenlm_perplexity_filter (CPU n-gram, reference-corpus signal)
#
# ── Deterministic filters (no model needed) ───────────────────────────────────
#   word_repeat_filter       – word n-gram repetition ratio
#   char_repeat_filter       – char n-gram repetition ratio
#   special_chars_filter     – special-character ratio
#   alphanumeric_filter      – alnum ratio
#   language_filter          – FastText language ID & confidence
#   flagged_words_filter     – offensive / blocked-word ratio
#   stopwords_filter         – stopword density (too low → code dump)
#   token_num_filter         – accurate token count via HF tokenizer
#   text_action_filter       – spaCy verb count (too few → static/passive)
#   kenlm_perplexity_filter  – n-gram PPL vs Wikipedia reference corpus
#   minhash_dedup            – MinHash LSH fuzzy near-duplicate removal
#
# ── Mappers (text normalization, applied before filtering) ────────────────────
#   fix_unicode              – ftfy unicode repair + NFC normalisation
#   remove_repeat_sentences  – exact duplicate sentence removal within a turn
#
# ── LLM-based filters (API mode → routes to our running sampler) ─────────────
#   llm_quality_filter       – accuracy/grammar/informativeness/coherence (1-5)
#   llm_difficulty_filter    – linguistic/conceptual/step complexity (1-5)
#   llm_condition_filter     – arbitrary natural-language yes/no condition
#   llm_task_relevance_filter– relevance to downstream eval task or dataset
#
# ── LLM-based filters (requires local GPU HF model) ──────────────────────────
#   ifd_filter               – Instruction Following Difficulty: L(A|Q)/L(A)
#                              higher → harder to follow → more informative
#
# ── Selectors (post-scoring, dataset-level) ───────────────────────────────────
#   topk_selector            – keep top-K rows by any computed stat field
from typing import Any, Dict, List, Optional, Union

from twinkle.preprocessor import Preprocessor


def _get_text(row: Dict[str, Any], role: str = 'assistant') -> str:
    """Concatenate all turns for a given role from messages."""
    parts = []
    for msg in row.get('messages') or []:
        if msg.get('role') == role:
            content = msg.get('content') or ''
            if isinstance(content, list):  # multimodal blocks
                content = ' '.join(b.get('text', '') for b in content if isinstance(b, dict))
            parts.append(str(content))
    return ' '.join(parts)


def _dj_dataset(texts: List[str]):
    """Wrap a list of strings into a Data-Juicer NestedDataset."""
    from data_juicer.core.data import NestedDataset
    from data_juicer.utils.constant import Fields
    import datasets
    ds = datasets.Dataset.from_dict({'text': texts})
    ds = ds.map(lambda x: {Fields.stats: {}, Fields.meta: {}}, batched=False)
    return NestedDataset(ds)


def _keep_mask(op, texts: List[str]) -> List[bool]:
    """Run a DJ Filter op and return a boolean keep-mask."""
    nd = _dj_dataset(texts)
    nd = op.compute_stats(nd)
    # process returns an iterable of booleans aligned with nd
    return list(op.process(nd))


class DataJuicerPreprocessor(Preprocessor):
    """Thin wrapper that exposes individual Data-Juicer filter ops
    as Preprocessor-compatible filter methods.

    All public methods accept and return List[Dict] (row-level).
    Use __call__ to invoke the full default pipeline.
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = self.word_repeat_filter(rows)
        rows = self.char_repeat_filter(rows)
        rows = self.special_chars_filter(rows)
        rows = self.alphanumeric_filter(rows)
        rows = self.map_row_to_col(rows)
        return rows

    # ── Repetition (replaces repeat.py) ───────────────────────────────────────

    def word_repeat_filter(
        self,
        rows: List[Dict[str, Any]],
        rep_len: int = 10,
        max_ratio: float = 0.4,
        role: str = 'assistant',
    ) -> List[Dict[str, Any]]:
        """Filter rows where word-level n-gram repetition ratio > max_ratio."""
        from data_juicer.ops.filter import WordRepetitionFilter
        op = WordRepetitionFilter(rep_len=rep_len, min_ratio=0.0, max_ratio=max_ratio)
        texts = [_get_text(r, role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]

    def char_repeat_filter(
        self,
        rows: List[Dict[str, Any]],
        rep_len: int = 10,
        max_ratio: float = 0.4,
        role: str = 'assistant',
    ) -> List[Dict[str, Any]]:
        """Filter rows where char-level n-gram repetition ratio > max_ratio."""
        from data_juicer.ops.filter import CharacterRepetitionFilter
        op = CharacterRepetitionFilter(rep_len=rep_len, min_ratio=0.0, max_ratio=max_ratio)
        texts = [_get_text(r, role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]

    # ── Character-level quality (complements token_soup.py) ───────────────────

    def special_chars_filter(
        self,
        rows: List[Dict[str, Any]],
        max_ratio: float = 0.25,
        role: str = 'assistant',
    ) -> List[Dict[str, Any]]:
        """Filter rows whose special-character ratio exceeds max_ratio."""
        from data_juicer.ops.filter import SpecialCharactersFilter
        op = SpecialCharactersFilter(min_ratio=0.0, max_ratio=max_ratio)
        texts = [_get_text(r, role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]

    def alphanumeric_filter(
        self,
        rows: List[Dict[str, Any]],
        min_ratio: float = 0.25,
        role: str = 'assistant',
    ) -> List[Dict[str, Any]]:
        """Filter rows whose alphanumeric-char ratio is below min_ratio."""
        from data_juicer.ops.filter import AlphanumericFilter
        op = AlphanumericFilter(tokenization=False, min_ratio=min_ratio)
        texts = [_get_text(r, role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]

    # ── Language ID (new capability) ──────────────────────────────────────────

    def language_filter(
        self,
        rows: List[Dict[str, Any]],
        lang: Union[str, List[str]] = '',
        min_score: float = 0.7,
        role: str = 'assistant',
    ) -> List[Dict[str, Any]]:
        """Keep rows whose detected language matches lang with confidence >= min_score.

        If lang is empty string, filter only on confidence (any language).
        """
        from data_juicer.ops.filter import LanguageIDScoreFilter
        op = LanguageIDScoreFilter(lang=lang, min_score=min_score)
        texts = [_get_text(r, role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]

    # ── Flagged words / offensive content (new capability) ────────────────────

    def flagged_words_filter(
        self,
        rows: List[Dict[str, Any]],
        lang: str = 'en',
        max_ratio: float = 0.045,
        role: str = 'assistant',
    ) -> List[Dict[str, Any]]:
        """Filter rows exceeding the flagged-word ratio threshold."""
        from data_juicer.ops.filter import FlaggedWordsFilter
        op = FlaggedWordsFilter(lang=lang, min_ratio=0.0, max_ratio=max_ratio)
        texts = [_get_text(r, role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]

    # ── Stopword ratio (new capability) ───────────────────────────────────────

    def stopwords_filter(
        self,
        rows: List[Dict[str, Any]],
        lang: str = 'en',
        min_ratio: float = 0.1,
        max_ratio: float = 1.0,
        role: str = 'assistant',
    ) -> List[Dict[str, Any]]:
        """Filter by stopword ratio.

        Too low (< 0.1) → likely code dump or gibberish.
        Too high → low-density filler text.
        """
        from data_juicer.ops.filter import StopWordsFilter
        op = StopWordsFilter(lang=lang, min_ratio=min_ratio, max_ratio=max_ratio)
        texts = [_get_text(r, role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]

    # ── KenLM perplexity (CPU, reference-corpus signal) ───────────────────────

    def kenlm_perplexity_filter(
        self,
        rows: List[Dict[str, Any]],
        lang: str = 'en',
        min_ppl: float = 0,
        max_ppl: float = 1500,
        role: str = 'assistant',
    ) -> List[Dict[str, Any]]:
        """Filter by KenLM perplexity (n-gram LM trained on Wikipedia).

        PPL too high → text deviates from clean reference corpus.
        Complements vLLM-based PerplexityFilter (which measures fit to
        the *current training model* rather than a reference corpus).
        """
        from data_juicer.ops.filter import PerplexityFilter as KenLMPPLFilter
        op = KenLMPPLFilter(lang=lang, min_ppl=min_ppl, max_ppl=max_ppl)
        texts = [_get_text(r, role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]

    # ── Near-duplicate removal ───────────────────────────────────────────────────

    def minhash_dedup(
        self,
        rows: List[Dict[str, Any]],
        tokenization: str = 'character',
        window_size: int = 5,
        num_permutations: int = 256,
        jaccard_threshold: float = 0.7,
        role: str = 'assistant',
    ) -> List[Dict[str, Any]]:
        """Remove near-duplicate rows via MinHash LSH.

        jaccard_threshold: rows with Jaccard similarity above this are duplicates.
        """
        from data_juicer.ops.deduplicator import DocumentMinhashDeduplicator
        from data_juicer.core.data import NestedDataset
        from data_juicer.utils.constant import Fields
        import datasets

        texts = [_get_text(r, role) for r in rows]
        ds = datasets.Dataset.from_dict({'text': texts})
        ds = ds.map(lambda x: {Fields.stats: {}, Fields.meta: {}}, batched=False)
        nd = NestedDataset(ds)

        op = DocumentMinhashDeduplicator(
            tokenization=tokenization,
            window_size=window_size,
            num_permutations=num_permutations,
            jaccard_threshold=jaccard_threshold,
        )
        nd = op.run(nd)
        keep_texts = set(nd['text'])
        # preserve original row order; drop duplicates
        seen, result = set(), []
        for r, t in zip(rows, texts):
            if t in keep_texts and t not in seen:
                seen.add(t)
                result.append(r)
        return result

    # ── Deterministic filters (continued) ───────────────────────────────────────

    def token_num_filter(
        self,
        rows: List[Dict[str, Any]],
        hf_tokenizer: str = 'Qwen/Qwen2.5-0.5B',
        min_num: int = 10,
        max_num: int = 8192,
        role: str = 'assistant',
    ) -> List[Dict[str, Any]]:
        """Filter by actual token count (more accurate than character count).

        Catches responses that are too short (boilerplate) or too long (bloat).
        """
        from data_juicer.ops.filter import TokenNumFilter
        op = TokenNumFilter(hf_tokenizer=hf_tokenizer, min_num=min_num, max_num=max_num)
        texts = [_get_text(r, role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]

    def text_action_filter(
        self,
        rows: List[Dict[str, Any]],
        lang: str = 'en',
        min_action_num: int = 1,
        role: str = 'assistant',
    ) -> List[Dict[str, Any]]:
        """Filter responses with fewer than min_action_num verbs (spaCy).

        Responses with near-zero verb count are typically passive acknowledgements
        or non-answers ('OK.', 'Sure!', etc.) that slip through simple length checks.
        lang: 'en' or 'zh'.
        """
        from data_juicer.ops.filter import TextActionFilter
        op = TextActionFilter(lang=lang, min_action_num=min_action_num)
        texts = [_get_text(r, role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]

    # ── Mappers (text normalization / cleaning) ─────────────────────────────────

    def fix_unicode(
        self,
        rows: List[Dict[str, Any]],
        normalization: str = 'NFC',
        role: str = 'assistant',
    ) -> List[Dict[str, Any]]:
        """Repair mojibake / encoding errors and NFC-normalise assistant text (ftfy).

        Run this BEFORE any filter that inspects character content.
        """
        from data_juicer.ops.mapper import FixUnicodeMapper
        op = FixUnicodeMapper(normalization=normalization)
        for row in rows:
            for msg in row.get('messages') or []:
                if msg.get('role') == role:
                    content = msg.get('content') or ''
                    if isinstance(content, str):
                        nd = _dj_dataset([content])
                        nd = op.run(nd)
                        msg['content'] = nd['text'][0]
        return rows

    def remove_repeat_sentences(
        self,
        rows: List[Dict[str, Any]],
        lowercase: bool = False,
        ignore_special_character: bool = True,
        role: str = 'assistant',
    ) -> List[Dict[str, Any]]:
        """Remove verbatim duplicate sentences within each assistant turn.

        Supports CJK sentence splitting (\u3002！？) and optional case/char normalisation.
        Does not remove cross-turn repetitions (use word_repeat_filter for that).
        """
        from data_juicer.ops.mapper import RemoveRepeatSentencesMapper
        op = RemoveRepeatSentencesMapper(
            lowercase=lowercase,
            ignore_special_character=ignore_special_character,
        )
        for row in rows:
            for msg in row.get('messages') or []:
                if msg.get('role') == role:
                    content = msg.get('content') or ''
                    if isinstance(content, str):
                        nd = _dj_dataset([content])
                        nd = op.run(nd)
                        msg['content'] = nd['text'][0]
        return rows

    # ── LLM-based filters (API mode → route to our sampler) ──────────────────────

    def llm_quality_filter(
        self,
        rows: List[Dict[str, Any]],
        api_endpoint: str,
        model: str = 'default',
        min_score: float = 0.5,
        role: str = 'assistant',
    ) -> List[Dict[str, Any]]:
        """Filter by LLM quality score (accuracy/grammar/informativeness/coherence).

        api_endpoint: URL of our sampler's /v1/chat/completions, e.g.
            'http://localhost:8000/v1/chat/completions'
        min_score: normalised 0-1 threshold (each dim is 1-5; avg / 5).
        """
        from data_juicer.ops.filter import LLMQualityScoreFilter
        op = LLMQualityScoreFilter(
            api_or_hf_model=model,
            api_endpoint=api_endpoint,
            min_score=min_score,
        )
        texts = [_get_text(r, role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]

    def llm_difficulty_filter(
        self,
        rows: List[Dict[str, Any]],
        api_endpoint: str,
        model: str = 'default',
        min_score: float = 0.4,
        max_score: float = 1.0,
        role: str = 'user',
    ) -> List[Dict[str, Any]]:
        """Filter by LLM difficulty score (linguistic/conceptual/step complexity).

        Applied to the user turn by default.
        Useful for curriculum: keep medium-to-hard queries only.
        """
        from data_juicer.ops.filter import LLMDifficultyScoreFilter
        op = LLMDifficultyScoreFilter(
            api_or_hf_model=model,
            api_endpoint=api_endpoint,
            min_score=min_score,
            max_score=max_score,
        )
        texts = [_get_text(r, role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]

    def llm_condition_filter(
        self,
        rows: List[Dict[str, Any]],
        condition: str,
        api_endpoint: str,
        model: str = 'default',
        role: str = 'assistant',
    ) -> List[Dict[str, Any]]:
        """Filter by an arbitrary natural-language yes/no condition (LLM judge).

        Examples:
            condition='the response is structured with clear sections'
            condition='the answer cites at least one source or reference'
            condition='the response is in the same language as the question'
        """
        from data_juicer.ops.filter import LLMConditionFilter
        op = LLMConditionFilter(
            condition=condition,
            api_or_hf_model=model,
            api_endpoint=api_endpoint,
        )
        texts = [_get_text(r, role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]

    def llm_task_relevance_filter(
        self,
        rows: List[Dict[str, Any]],
        api_endpoint: str,
        task_desc: Optional[str] = None,
        valid_examples: Optional[List[Dict[str, Any]]] = None,
        model: str = 'default',
        min_score: float = 0.5,
        role: str = 'assistant',
    ) -> List[Dict[str, Any]]:
        """Filter by relevance to a downstream task or validation dataset.

        Provide task_desc (string) and/or valid_examples (list of {text: ...} dicts)
        to characterise the target domain. High score = likely to help downstream.
        """
        from data_juicer.ops.filter import LLMTaskRelevanceFilter
        op = LLMTaskRelevanceFilter(
            api_or_hf_model=model,
            api_endpoint=api_endpoint,
            min_score=min_score,
            valid_dataset=valid_examples,
            task_desc=task_desc,
        )
        texts = [_get_text(r, role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]

    # ── LLM-based filters (requires local HF model on GPU) ───────────────────────

    def ifd_filter(
        self,
        rows: List[Dict[str, Any]],
        hf_model: str,
        min_score: float = 0.5,
        max_score: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Filter by Instruction Following Difficulty (IFD) score.

        IFD = L(A | Q) / L(A)  where L is the model's per-token loss.
        Higher IFD → the query provides more task-constraining signal →
        more informative training example. (Paper: https://arxiv.org/abs/2308.12032)

        Requires a local HF model loaded on GPU (not API mode).
        Typical range: keep 0.5-2.0 (discard near-zero = trivial, >2 = noisy).
        """
        from data_juicer.ops.filter import InstructionFollowingDifficultyFilter
        op = InstructionFollowingDifficultyFilter(
            hf_model=hf_model,
            min_score=min_score,
            max_score=max_score,
        )
        # IFD op works on {messages: [...]} samples directly
        nd = _dj_dataset([''])  # placeholder; op reads 'messages' field
        # build per-row samples for single-sample processing
        results = []
        for row in rows:
            sample = {'messages': row.get('messages') or [], '__dj__stats__': {}, '__dj__meta__': {}}
            sample = op.compute_stats_single(sample)
            score = sample['__dj__stats__'].get('ifd_score', 1.0)
            if min_score <= score <= max_score:
                results.append(row)
        return results

    # ── Selector (dataset-level, run after scoring) ──────────────────────────────

    def topk_selector(
        self,
        rows: List[Dict[str, Any]],
        score_fn,
        topk: Optional[int] = None,
        top_ratio: Optional[float] = None,
        reverse: bool = True,
    ) -> List[Dict[str, Any]]:
        """Keep top-K rows by a caller-supplied scoring function.

        score_fn(row) -> float.  Rows are sorted descending (reverse=True)
        then the top topk / top_ratio fraction are returned.

        Example: keep top-20% by response length
            topk_selector(rows, score_fn=lambda r: len(_get_text(r)), top_ratio=0.2)

        Example: keep top-500 by LLM quality score stored in row['_quality']
            topk_selector(rows, score_fn=lambda r: r.get('_quality', 0), topk=500)
        """
        if not rows:
            return rows
        scored = [(score_fn(r), i) for i, r in enumerate(rows)]
        scored.sort(key=lambda x: x[0], reverse=reverse)

        n = len(rows)
        if topk is not None and top_ratio is not None:
            k = min(topk, int(n * top_ratio))
        elif topk is not None:
            k = topk
        elif top_ratio is not None:
            k = int(n * top_ratio)
        else:
            return rows
        k = max(1, min(k, n))

        keep_indices = {i for _, i in scored[:k]}
        return [r for i, r in enumerate(rows) if i in keep_indices]
