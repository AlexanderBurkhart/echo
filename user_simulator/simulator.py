"""Scores one output using frozen measurements and private user rules."""

import collections
from collections.abc import Sequence
import datetime
import pathlib
import time
import uuid

from conversation import history
from experiment_logging import records
from experiment_logging import runs
from user_simulator import config
from user_simulator import engine
from user_simulator.data import score_records
from user_simulator.data import judgments
from user_simulator.judge import judge as judge_interface
from user_simulator.judge import judging
from user_simulator.judge import prompts
from user_simulator.judge import response_schema
from user_simulator.judge.ratings import ratings as rating_scale


TASK_CACHE_DIRECTORY = "task_cache"
OUTPUT_CACHE_DIRECTORY = "output_cache"
MAX_LM_JUDGMENT_ATTEMPTS = 3


class JudgeError(RuntimeError):
    """A scoring request failed to obtain a usable LM judgment.

    Attributes:
        diagnostics: Path to the saved score record describing the failure.
    """

    def __init__(self, message: str, diagnostics: pathlib.Path) -> None:
        """Initializes an error with its private diagnostics path."""
        super().__init__(message)
        self.diagnostics = diagnostics


def _get_timestamp() -> str:
    """Returns the current UTC time in ISO format."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class Simulator:
    """A single-process evaluator that returns only a numerical score.

    Task conditions share one LM call; each output criterion has its own call.
    Successful LM assessments are cached across users and restarts. Scores
    are calculated from the measurements and each user's private configuration.
    Keep private run artifacts inaccessible to learners.
    This class is not thread-safe and does not enforce operating-system access.
    """

    def __init__(
        self,
        users: Sequence[config.UserConfig],
        judge: judge_interface.Judge,
        *,
        component: str = "user_simulator",
    ):
        """Initializes fixed private users and a versioned measurement cache.

        Args:
            users: Immutable private user configurations, indexed by ID.
            judge: LM judge using a fresh context for each prompt.
            component: Namespace for this evaluator within the shared run.
        """
        user_id_counts = collections.Counter(user.user_id for user in users)
        if any(count > 1 for count in user_id_counts.values()):
            raise ValueError("Duplicate user ID")
        self._users = {user.user_id: user for user in users}
        self._judge = judge
        self._identity = judge.identity
        self._component = component

    def _request_lm_judgment(
        self,
        kind: judgments.JudgmentKind,
        sources: list[str],
        request_dir: pathlib.Path,
        score_record: score_records.ScoreRecord,
    ) -> None:
        """Stores the LM judgment and each call outcome in score_record."""
        score_path = request_dir / "score.json"
        criteria = score_record.user_config.rating_criteria
        criteria_per_call = (
            tuple((item,) for item in criteria)
            if kind is judgments.JudgmentKind.OUTPUT
            else (criteria,)
        )
        combined = judgments.LMJudgment(
            ratings={}, source_record=str(score_path.resolve()),
        )
        for call_criteria in criteria_per_call:
            request_prompt = score_record.task_prompt
            if kind is judgments.JudgmentKind.OUTPUT:
                # Each output criterion has its own prompt, saved on its call.
                request_prompt = prompts.build_criterion_prompt(
                    score_record.history, score_record.output, call_criteria[0],
                )
            self._request_lm_judgment_with_retries(
                kind=kind, sources=sources, criteria=call_criteria,
                request_prompt=request_prompt, score_path=score_path,
                score_record=score_record, combined=combined,
            )
        if kind is judgments.JudgmentKind.TASK:
            score_record.task_judgment = combined
        else:
            score_record.output_judgment = combined

    def _request_lm_judgment_with_retries(
        self,
        *,
        kind: judgments.JudgmentKind,
        sources: list[str],
        criteria: tuple[rating_scale.RatingAnchoredCriterion, ...],
        request_prompt: str,
        score_path: pathlib.Path,
        score_record: score_records.ScoreRecord,
        combined: judgments.LMJudgment,
    ) -> None:
        """Records attempts and adds a valid response to the combined judgment.

        Retries malformed LM responses up to three total attempts. Runtime
        errors fail immediately. Every attempt remains in the score record.
        """
        calls = (
            score_record.task_lm_calls
            if kind is judgments.JudgmentKind.TASK
            else score_record.output_lm_calls
        )
        schema = response_schema.build_response_schema(kind, criteria)
        original_prompt = request_prompt
        for _ in range(MAX_LM_JUDGMENT_ATTEMPTS):
            started = time.perf_counter()
            call = score_records.LMCallResult(
                prompt=request_prompt, response_schema=schema,
            )
            try:
                call.raw = self._judge.complete(
                    request_prompt, response_schema=schema,
                )
                parsed = judging.parse_lm_judgment(
                    call.raw, kind, sources, criteria,
                    require_feedback=(
                        kind is judgments.JudgmentKind.OUTPUT
                    ),
                )
                combined.ratings.update(parsed.ratings)
                combined.quantities.update(parsed.quantities)
                break
            except judging.MalformedLMJudgmentError as error:
                call.error = str(error)
                request_prompt = prompts.build_retry_prompt(
                    original_prompt, call.raw, call.error,
                )
            except Exception as error:
                call.error = f"{type(error).__name__}: {error}"
                raise JudgeError(
                    "Judge runtime failed; pause for debugging", score_path
                ) from error
            finally:
                call.elapsed_seconds = time.perf_counter() - started
                calls.append(call)
        else:
            raise JudgeError(
                f"Malformed LM judgment after {MAX_LM_JUDGMENT_ATTEMPTS} "
                "attempts; pause", score_path,
            )

    def _get_or_request_lm_assessment(
        self,
        *,
        kind: judgments.JudgmentKind,
        sources: list[str],
        request_dir: pathlib.Path,
        cache_path: pathlib.Path,
        run_root: str | pathlib.Path,
        use_cache: bool,
        score_record: score_records.ScoreRecord,
    ) -> None:
        """Stores LM ratings and cache details in score_record."""
        if kind is judgments.JudgmentKind.TASK:
            score_record.task_cache_hit = False
        else:
            score_record.output_cache_hit = False
        if use_cache and cache_path.exists():
            raw = None
            try:
                raw = cache_path.read_text(encoding="utf-8")
                cached = judging.decode_lm_json(raw)
                # Remove provenance before validating the cached LM response.
                source_record = cached.pop("source_record", None)
                if not isinstance(source_record, str) or not source_record:
                    raise ValueError("Cached LM judgment needs source_record")
                cached.pop("_record", None)
                parsed = judging.parse_lm_judgment(
                    records.serialize_canonical_json(cached),
                    kind,
                    sources,
                    score_record.user_config.rating_criteria,
                )
            except ValueError as error:
                cache_error = f"{type(error).__name__}: {error}"
                if kind is judgments.JudgmentKind.TASK:
                    score_record.task_cache_error = cache_error
                    score_record.task_invalid_cache = raw
                else:
                    score_record.output_cache_error = cache_error
                    score_record.output_invalid_cache = raw
            else:
                parsed.source_record = source_record
                if kind is judgments.JudgmentKind.TASK:
                    score_record.task_judgment = parsed
                    score_record.task_cache_hit = True
                else:
                    score_record.output_judgment = parsed
                    score_record.output_cache_hit = True
                return
        self._request_lm_judgment(
            kind=kind,
            sources=sources,
            request_dir=request_dir,
            score_record=score_record,
        )
        if use_cache:
            judgment = (
                score_record.task_judgment
                if kind is judgments.JudgmentKind.TASK
                else score_record.output_judgment
            )
            runs.write_record(
                run_root, self._component, cache_path, judgment
            )

    def score(
        self,
        *,
        run_root: str | pathlib.Path,
        user_id: str,
        conversation_history: history.ConversationHistory,
        output: str,
        use_cache: bool = True,
    ) -> float:
        """Gets LM measurements, then calculates the preference score in code.

        Args:
            run_root: Run directory for shared logs and cached LM ratings.
            user_id: ID to look up in the fixed user snapshot.
            conversation_history: Visible messages supplied by the caller,
                without private configuration data.
            output: One complete, readable generated output or serialized trace.
            use_cache: Reuses matching task and output LM judgments. False
                bypasses all cache reads and writes for repeatability audits.

        Returns:
            A value in [0, 100]. Objective correctness is separate.

        Raises:
            JudgeError: Runtime failure or malformed-attempt limit.
            KeyError: The requested user ID is not configured.
            ValueError: Output is empty or the frozen judge identity changed.
            OSError: Required diagnostics or cache files cannot be saved.
        """
        started = time.perf_counter()
        if self._judge.identity != self._identity:
            raise ValueError("Judge identity changed during a frozen run")
        if not isinstance(output, str) or not output.strip():
            raise ValueError("Output must be a nonempty readable string")
        user = self._users[user_id]
        prompt_hash = prompts.compute_prompt_fingerprint(
            user.rating_criteria,
        )
        namespace = records.compute_digest(score_records.JudgeNamespace(
            judge=self._identity, prompts=prompt_hash,
        ))
        root = runs.get_component_directory(run_root, self._component)
        sources = [message.content for message in conversation_history.messages]
        task_key = records.compute_digest(score_records.LMCacheKey(
            namespace=namespace, history=conversation_history,
        ))
        output_key = records.compute_digest(score_records.LMCacheKey(
            namespace=namespace, history=conversation_history,
            output=output,
        ))
        task_cache_path = (
            root / TASK_CACHE_DIRECTORY / f"{task_key}.json"
        )
        output_cache_path = (
            root / OUTPUT_CACHE_DIRECTORY / f"{output_key}.json"
        )
        request_dir = root / runs.REQUESTS_DIRECTORY / uuid.uuid4().hex
        score_record = score_records.ScoreRecord(
            user_id=user_id,
            history=conversation_history,
            output=output,
            judge=self._identity,
            prompt_hash=prompt_hash,
            user_config=user,
            task_cache_key=task_key,
            output_cache_key=output_key,
            timestamp=_get_timestamp(),
            task_prompt=prompts.build_task_prompt(conversation_history),
        )
        try:
            # Reuse the LM's general task-condition judgments, which determine
            # the preference weights selected for this user.
            self._get_or_request_lm_assessment(
                kind=judgments.JudgmentKind.TASK,
                sources=sources,
                request_dir=request_dir,
                cache_path=task_cache_path,
                run_root=run_root,
                use_cache=use_cache,
                score_record=score_record,
            )
            # Reuse the LM's output ratings against the fixed preference
            # rubrics; code combines them with the user's weights below.
            self._get_or_request_lm_assessment(
                kind=judgments.JudgmentKind.OUTPUT,
                sources=sources + [output],
                request_dir=request_dir,
                cache_path=output_cache_path,
                run_root=run_root,
                use_cache=use_cache,
                score_record=score_record,
            )
            # Task ratings select this user's preference weights; code
            # combines those weights with the LM's output ratings to calculate
            # the user's preference score deterministically.
            result = engine.calculate_preference_score(
                user,
                score_record.task_judgment.ratings,
                score_record.output_judgment.ratings,
            )
            score_record.result = result
            score_record.status = score_records.ScoreStatus.OK
        except Exception as error:
            # Save the failure in score_record, then preserve the exception.
            score_record.status = score_records.ScoreStatus.FAILURE
            score_record.error = f"{type(error).__name__}: {error}"
            raise
        finally:
            score_record.elapsed_seconds = time.perf_counter() - started
            runs.write_record(
                run_root, self._component, request_dir / "score.json",
                score_record,
            )
        return result.score
