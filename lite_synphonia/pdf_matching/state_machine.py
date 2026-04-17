from __future__ import annotations

from dataclasses import dataclass, field

from .scorer import PageScore


@dataclass(frozen=True)
class MatchingConfig:
    query_max_seconds: float = 18.0
    query_max_segments: int = 8
    switch_margin: float = 0.032
    low_confidence_threshold: float = 0.12
    min_page_dwell_seconds: float = 1.5
    cooldown_seconds: float = 1.5
    confirmations_required: int = 1
    current_page_bonus: float = 0.04
    next_page_bonus: float = 0.025
    previous_page_bonus: float = 0.008
    nearby_range_bonus: float = 0.004
    far_page_penalty: float = 0.015


@dataclass(frozen=True)
class MatchDecision:
    assigned_page: int | None
    candidate_page: int | None
    confidence: float
    switched: bool
    reason: str
    page_scores: list[dict[str, float | int]]


@dataclass
class PageTimelineEntry:
    page_index: int
    start_time: float
    end_time: float
    confidence: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "page_index": self.page_index,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration": round(self.duration, 3),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class PageMatcherState:
    config: MatchingConfig = field(default_factory=MatchingConfig)
    current_page: int | None = None
    current_page_start_time: float | None = None
    current_page_confidence: float = 0.0
    last_switch_time: float | None = None
    pending_candidate: int | None = None
    pending_count: int = 0
    low_confidence_segments: int = 0
    timeline: list[PageTimelineEntry] = field(default_factory=list)

    def update(self, *, start_time: float, end_time: float, page_scores: list[PageScore]) -> MatchDecision:
        if not page_scores:
            self.low_confidence_segments += 1
            return MatchDecision(None, None, 0.0, False, "no_scores", [])

        adjusted_scores = self._apply_page_priors(page_scores)
        best = adjusted_scores[0]
        best_raw = self._find_score(page_scores, best.page_index)
        current_raw = self._find_score(page_scores, self.current_page)
        decision_payload = [self._score_to_dict(item) for item in adjusted_scores[:5]]

        raw_best = page_scores[0]
        current_score = current_raw.score if current_raw is not None else -1.0
        if (
            self.current_page is not None
            and best.page_index == self.current_page
            and raw_best.page_index != self.current_page
            and raw_best.score >= current_score + self.config.switch_margin
        ):
            best = self._find_score(adjusted_scores, raw_best.page_index) or raw_best
            best_raw = raw_best

        if best.score < self.config.low_confidence_threshold:
            self.low_confidence_segments += 1
            return MatchDecision(
                assigned_page=self.current_page,
                candidate_page=best.page_index,
                confidence=max(best.score, 0.0),
                switched=False,
                reason="low_confidence",
                page_scores=decision_payload,
            )

        if self.current_page is None:
            self._start_page(best.page_index, start_time, best.score)
            return MatchDecision(
                assigned_page=best.page_index,
                candidate_page=best.page_index,
                confidence=best.score,
                switched=True,
                reason="initial_page_lock",
                page_scores=decision_payload,
            )

        if best.page_index == self.current_page:
            self.pending_candidate = None
            self.pending_count = 0
            self.current_page_confidence = max(self.current_page_confidence, best.score)
            return MatchDecision(
                assigned_page=self.current_page,
                candidate_page=best.page_index,
                confidence=best.score,
                switched=False,
                reason="stay_current_page",
                page_scores=decision_payload,
            )

        dwell_seconds = (
            0.0
            if self.current_page_start_time is None
            else max(0.0, start_time - self.current_page_start_time)
        )
        if dwell_seconds < self.config.min_page_dwell_seconds:
            return MatchDecision(
                assigned_page=self.current_page,
                candidate_page=best.page_index,
                confidence=best.score,
                switched=False,
                reason="min_dwell_not_met",
                page_scores=decision_payload,
            )

        if (
            self.last_switch_time is not None
            and start_time - self.last_switch_time < self.config.cooldown_seconds
            and best_raw is not None
            and best_raw.score < current_score + self.config.switch_margin * 1.5
        ):
            return MatchDecision(
                assigned_page=self.current_page,
                candidate_page=best.page_index,
                confidence=best.score,
                switched=False,
                reason="cooldown_active",
                page_scores=decision_payload,
            )

        if best_raw is not None and best_raw.score < current_score + self.config.switch_margin:
            self.pending_candidate = None
            self.pending_count = 0
            return MatchDecision(
                assigned_page=self.current_page,
                candidate_page=best.page_index,
                confidence=best.score,
                switched=False,
                reason="switch_margin_not_met",
                page_scores=decision_payload,
            )

        if self.pending_candidate == best.page_index:
            self.pending_count += 1
        else:
            self.pending_candidate = best.page_index
            self.pending_count = 1

        if self.pending_count < self.config.confirmations_required:
            return MatchDecision(
                assigned_page=self.current_page,
                candidate_page=best.page_index,
                confidence=best.score,
                switched=False,
                reason="awaiting_confirmation",
                page_scores=decision_payload,
            )

        self._close_current_page(start_time)
        self._start_page(best.page_index, start_time, best.score)
        self.last_switch_time = start_time
        self.pending_candidate = None
        self.pending_count = 0
        return MatchDecision(
            assigned_page=best.page_index,
            candidate_page=best.page_index,
            confidence=best.score,
            switched=True,
            reason="page_switch_confirmed",
            page_scores=decision_payload,
        )

    def finalize(self, end_time: float) -> list[PageTimelineEntry]:
        self._close_current_page(end_time)
        return list(self.timeline)

    def _apply_page_priors(self, page_scores: list[PageScore]) -> list[PageScore]:
        adjusted: list[PageScore] = []
        for score in page_scores:
            bonus = 0.0
            if self.current_page is not None:
                distance = score.page_index - self.current_page
                if distance == 0:
                    bonus += self.config.current_page_bonus
                elif distance == 1:
                    bonus += self.config.next_page_bonus
                elif distance == -1:
                    bonus += self.config.previous_page_bonus
                elif -1 <= distance <= 3:
                    bonus += self.config.nearby_range_bonus
                else:
                    bonus -= self.config.far_page_penalty
            adjusted.append(
                PageScore(
                    page_index=score.page_index,
                    score=score.score + bonus,
                    max_score=score.max_score,
                    mean_top_score=score.mean_top_score,
                    chunk_hits=score.chunk_hits,
                )
            )

        adjusted.sort(key=lambda item: (item.score, item.max_score), reverse=True)
        return adjusted

    def _start_page(self, page_index: int, start_time: float, confidence: float) -> None:
        self.current_page = page_index
        self.current_page_start_time = start_time
        self.current_page_confidence = confidence

    def _close_current_page(self, end_time: float) -> None:
        if self.current_page is None or self.current_page_start_time is None:
            return
        if end_time < self.current_page_start_time:
            end_time = self.current_page_start_time
        if self.timeline and self.timeline[-1].page_index == self.current_page:
            self.timeline[-1].end_time = end_time
            self.timeline[-1].confidence = max(
                self.timeline[-1].confidence,
                self.current_page_confidence,
            )
        else:
            self.timeline.append(
                PageTimelineEntry(
                    page_index=self.current_page,
                    start_time=self.current_page_start_time,
                    end_time=end_time,
                    confidence=self.current_page_confidence,
                )
            )

    def _find_score(self, page_scores: list[PageScore], page_index: int | None) -> PageScore | None:
        if page_index is None:
            return None
        for score in page_scores:
            if score.page_index == page_index:
                return score
        return None

    def _score_to_dict(self, score: PageScore) -> dict[str, float | int]:
        return {
            "page_index": score.page_index,
            "score": round(score.score, 4),
            "max_score": round(score.max_score, 4),
            "mean_top_score": round(score.mean_top_score, 4),
            "chunk_hits": score.chunk_hits,
        }
