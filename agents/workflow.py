import json
import operator
import os
import time
import uuid
from datetime import datetime
from typing import TypedDict, Literal, Any, Annotated, Optional
from agents.llm_factory import get_llm

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.errors import GraphInterrupt
from langchain_core.messages import HumanMessage, AIMessage

from agents.orchestrator import (
    create_orchestrator_node, load_student_kb, read_skb_database, SKB_FILE_PATH,
    DIAGNOSIS_NODE, TUTOR_NODE, ASSESSMENT_NODE,
)
from agents.diagnosis_agent import DiagnosisAgent
from agents.tutor_agent import create_tutor_node, clean_tutor_output
from agents.assessment_agent import create_assessment_node, load_all_mcqs
from agents.error_analysis_agent import ErrorAnalysisAgent
from agents.root_cause_agent import (
    DEFAULT_METHOD, DEFAULT_PRESET, METHOD_LABELS, RootCauseAgent, format_hypotheses, format_root_cause,
    resolve_weights,
    summarize_comparison,
)
from agents.pedagogy_controller import record_assessment_outcome, resolve_topic_id, update_subtopic_mastery
from agents.struggle_tracker import recent_learning_topics
from agents.chat_memory import history_messages, lesson_entry, quiz_entry, remember

# --- Configuration & File Path Constants ---
# Every start()/resume() call appends its agent trace here (one JSON line per call, one file
# per thread), so a study session can be reviewed after the browser is closed.
TRACE_DIR = "data/traces"
MAX_ITERATIONS = 3
QUIZ_SIZE = 10


class AgentState(TypedDict):
    # inputs
    student_id: str
    user_input: str
    chat_history: list
    messages: list                  
    student_answers: dict
    # routing / metadata from orchestrator
    topic_focus: list
    orchestrator_instruction: str
    orchestrator_reasoning: str
    conversation_mode: str          # "new_request", "follow_up" (answer to the tutor's question),
                                    # "remediation" or "retest" (app buttons after a quiz)
    tutor_question: str             # the question being answered, for a follow-up
    turn_analysis: Optional[dict]   # turn judge: answer/new question, verdict, struggle flags
    forced_route: Optional[dict]    # remediation / re-test chosen by an app button
    quiz_subtopics: list            # re-test: the subtopics the quiz must cover
    # per-round data
    student_kb: Optional[dict]
    active_quiz_questions: list
    graded_details: list
    wrong_problems: list
    score_history: list
    # outputs
    round_log: Annotated[list, operator.add]
    response_output: Any
    last_active_agent: Optional[str]


class AdaptiveTutorWorkflow:
    # Root-cause configuration; app.py sets these from the sidebar before each run.
    root_cause_method = DEFAULT_METHOD
    root_cause_preset = DEFAULT_PRESET
    root_cause_weights = None            # custom weights; None = use the preset

    def __init__(self, db_filepath: str = SKB_FILE_PATH):
        self.db_filepath = db_filepath
        self.diagnosis_agent = DiagnosisAgent(db_filepath=db_filepath)

        self.orchestrator_node_fn = create_orchestrator_node()
        self.tutor_node_fn = create_tutor_node(enable_kb_context=True)
        self.assessment_node_fn = create_assessment_node()
        self.error_analysis_agent = ErrorAnalysisAgent()
        self.root_cause_agent = RootCauseAgent()

        # Steps executed during the current start()/resume() call, returned to the UI.
        self._trace = []

        self.checkpointer = MemorySaver()
        self.app = self._build_graph()

    # ------------------------------------------------------------------ trace
    def _log(self, agent: str, title: str, inputs: dict, outputs: dict, started: float = None) -> None:
        """Record one agent step (who ran, with which input, and what it produced)."""
        if not hasattr(self, "_trace"):
            self._trace = []
        self._trace.append({
            "agent": agent,
            "title": title,
            "input": inputs,
            "output": outputs,
            "duration_ms": round((time.perf_counter() - started) * 1000) if started else None,
        })

    @staticmethod
    def _kb_summary(student_kb: Optional[dict]) -> dict:
        if not student_kb:
            return {"registered": False}
        return {
            "registered": True,
            "topics": student_kb.get("topics", {}),
            "weaknesses": student_kb.get("weaknesses", []),
            "lessons": sum(e.get("event") == "lesson" for e in student_kb.get("pedagogy_history", [])),
            "quizzes": sum(e.get("event") == "assessment" for e in student_kb.get("pedagogy_history", [])),
            "learning_turns": len(student_kb.get("learning_conversations", [])),
        }

    # ------------------------------------------------------------------ nodes
    # The orchestrator routes by returning Command(goto=<node name>). The annotation lists the
    # possible targets so LangGraph knows the graph's edges (e.g. when drawing it).
    def orchestrator_node(self, state: AgentState) -> Command[Literal["diagnosis", "tutor_agent", "assessment_agent"]]:
        print("\n🔵 [Orchestrator] Analyzing student status and intent...")
        started = time.perf_counter()
        command = self.orchestrator_node_fn(state)
        update = command.update or {}
        self._log(
            "Orchestrator", f"Routed to {command.goto}",
            {
                "user_message": state.get("user_input"),
                "chat_history": [
                    f"{'student' if isinstance(m, HumanMessage) else 'tutor'}: {str(m.content)[:120]}"
                    for m in state.get("messages", [])[:-1]
                ],
                "recent_learning_topics": recent_learning_topics(state.get("student_kb")),
                "student_kb": self._kb_summary(state.get("student_kb")),
            },
            {
                "next_agent": command.goto,
                "topic_focus": update.get("topic_focus"),
                "instruction": update.get("orchestrator_instruction"),
                "reasoning": update.get("orchestrator_reasoning"),
                "turn_judge": update.get("turn_analysis"),
                "forced_route": state.get("forced_route"),
            },
            started,
        )
        return command

    def diagnosis_node(self, state: AgentState) -> dict:
        print("\n🟢 [Diagnosis] Administering diagnostic assessment test...")
        questions = self.diagnosis_agent.get_questions()
        try:
            answers = interrupt({
                "type": "diagnosis",
                "student_id": state["student_id"],
                "questions": questions,
            })
        except GraphInterrupt:
            self._log("Diagnosis Agent", "Paused: waiting for diagnostic answers",
                      {"student_id": state["student_id"]},
                      {"questions_sent": [q["problem_id"] for q in questions]})
            raise

        started = time.perf_counter()
        new_kb = self.diagnosis_agent.administer_test(state["student_id"], answers)
        skb_dict = new_kb.model_dump(mode="json")
        self._log(
            "Diagnosis Agent", "Graded diagnostic test and created the knowledge base",
            {"answers": answers},
            {"topics": skb_dict["topics"], "strengths": skb_dict["strengths"],
             "weaknesses": skb_dict["weaknesses"], "misconceptions": skb_dict["active_misconceptions"]},
            started,
        )

        return {
            "student_kb": skb_dict,
            "round_log": [{
                "type": "diagnosis_completed",
                "topics": skb_dict["topics"],
                "strengths": skb_dict["strengths"],
                "weaknesses": skb_dict["weaknesses"],
                "misconceptions": skb_dict["active_misconceptions"],
            }],
            "response_output": {
                "status": "diagnosis_completed",
                "message": "Diagnostic assessment completed successfully! You can now ask questions or take quizzes."
            },
            "last_active_agent": "diagnosis_agent"
        }

    def tutor_node(self, state: AgentState) -> dict:
        print(f"\n🟡 [Tutor] Handling explanation request for topic: {state.get('topic_focus')}")
        started = time.perf_counter()
        result = self.tutor_node_fn(state)
        response_text = result["messages"][-1].content
        student_kb = result.get("student_kb", state.get("student_kb"))
        topic_focus = state.get("topic_focus") or []
        if not isinstance(topic_focus, list):
            topic_focus = [str(topic_focus)]
        topic_id = resolve_topic_id(topic_focus[0] if topic_focus else "unknown")
        previous_strategy = next(
            (
                event.get("strategy")
                for event in reversed((state.get("student_kb") or {}).get("pedagogy_history", []))
                if event.get("event") == "lesson"
                and event.get("topic_id") == topic_id
            ),
            None,
        )
        strategy = result["pedagogy_strategy"]
        strategy_label = strategy.replace("_", " ").title()
        if previous_strategy is None:
            print(f"\n🧠 [Pedagogy] Selected teaching strategy: {strategy_label}")
        elif previous_strategy != strategy:
            previous_label = previous_strategy.replace("_", " ").title()
            print(
                f"\n🧠 [Pedagogy] Teaching strategy changed: "
                f"{previous_label} → {strategy_label}"
            )
        else:
            print(f"\n🧠 [Pedagogy] Continuing teaching strategy: {strategy_label}")
        print(f"   - Reason: {result['pedagogy_reason']}")

        self._log(
            "Pedagogy Controller", f"Strategy: {strategy_label}",
            {"topic_id": topic_id, "previous_strategy": previous_strategy,
             "topic_mastery": (state.get("student_kb") or {}).get("topics", {}).get(topic_id)},
            {"strategy": strategy, "changed": previous_strategy not in (None, strategy),
             "reason": result["pedagogy_reason"]},
        )
        logged_turn = ((student_kb or {}).get("learning_conversations") or [None])[-1]
        self._log(
            "Tutor Agent",
            "Evaluated the student's answer and continued the lesson"
            if result.get("conversation_mode") == "follow_up" else "Generated explanation",
            {"topic_focus": topic_focus, "instruction": state.get("orchestrator_instruction"),
             "strategy": strategy, "student_message": state.get("user_input"),
             "conversation_mode": result.get("conversation_mode"),
             "answering_tutor_question": state.get("tutor_question") or None},
            {"response_preview": response_text[:600] + ("…" if len(response_text) > 600 else ""),
             "answer_verdict": result.get("answer_verdict"),
             "verdict_source": result.get("verdict_source"),
             "learning_turn_logged": logged_turn},
            started,
        )

        if student_kb:
            # Keep this exchange in the chat memory (latest 3, summarised).
            student_kb = remember(student_kb, lesson_entry(
                state.get("user_input"), response_text, topic_id, strategy,
                result.get("conversation_mode"), result.get("answer_verdict"),
            ))
            self._save_student_kb(student_kb)

        return {
            "student_kb": student_kb,
            "round_log": [{
                "type": "explanation",
                "focus_topic": state.get("topic_focus"),
                "text": response_text,
                "pedagogy_strategy": result["pedagogy_strategy"],
                "pedagogy_reason": result["pedagogy_reason"],
            }],
            "response_output": {
                "status": "completed",
                "response_text": response_text,
                "pedagogy_strategy": result["pedagogy_strategy"],
                "pedagogy_reason": result["pedagogy_reason"],
                "conversation_mode": result.get("conversation_mode"),
                "answer_verdict": result.get("answer_verdict"),
                "verdict_source": result.get("verdict_source"),
                "remediation_for": (state.get("forced_route") or {}).get("quiz_id")
                if result.get("conversation_mode") == "remediation" else None,
            },
            "last_active_agent": "tutor_agent"
        }

    def assessment_quiz_node(self, state: AgentState) -> dict:
        print(f"\n🟣 [Assessment] Generating adaptive quiz for topic: {state.get('topic_focus')}")
        started = time.perf_counter()
        result = self.assessment_node_fn(state)
        intro_text = result["messages"][-1].content
        quiz_questions = result["active_quiz_questions"]
        # On resume LangGraph re-runs this node from the top; only log the selection the first time.
        if not getattr(self, "_resuming", False):
            self._log(
                "Assessment Agent", "Selected adaptive quiz questions",
                {"topic_focus": state.get("topic_focus"), "instruction": state.get("orchestrator_instruction")},
                {"question_ids": [q.get("prob_id", q.get("problem_id")) for q in quiz_questions], "intro": intro_text},
                started,
            )

        try:
            answers = interrupt({
                "type": "quiz",
                "topic_focus": state.get("topic_focus"),
                "intro_text": intro_text,
                "questions": quiz_questions,
            })
        except GraphInterrupt:
            self._log("Assessment Agent", "Paused: waiting for quiz answers", {}, {"questions": len(quiz_questions)})
            raise

        # When the student submits, LangGraph runs this node again from the top, and the random
        # selection above picks a new set of questions. Grade the questions the student actually
        # answered (the ids in `answers`), looked up in the question bank.
        bank = {q.get("prob_id", q.get("problem_id")): q for q in load_all_mcqs()}
        answered = [bank[pid] for pid in (answers or {}) if pid in bank]
        if answered:
            quiz_questions = answered

        correct_count = 0
        graded_details = []
        wrong_problems = []

        for q in quiz_questions:
            pid = q.get("prob_id", q.get("problem_id"))
            correct_val = q.get("correct_option")
            student_val = answers.get(pid)
            is_correct = (student_val == correct_val)
            
            if is_correct:
                correct_count += 1
            else:
                wrong_problems.append(q)
                
            graded_details.append({"prob_id": pid, "is_correct": is_correct})

        total = len(quiz_questions)
        accuracy = round(correct_count / total, 2) if total > 0 else 0.0
        # A quiz can cover several topics (e.g. the topics of the last 3 tutor conversations).
        topic_focus = state.get("topic_focus") or ["unknown"]
        if not isinstance(topic_focus, list):
            topic_focus = [str(topic_focus)]
        topic_ids = list(dict.fromkeys(resolve_topic_id(t) for t in topic_focus))
        self._log(
            "Assessment Agent", f"Received answers and graded quiz: {correct_count}/{len(quiz_questions)}",
            {"answers": answers},
            {"graded": graded_details, "accuracy": round(correct_count / len(quiz_questions), 2) if quiz_questions else 0.0},
        )

        started = time.perf_counter()
        # Each question is credited to the first quiz topic it covers, so a question tagged with
        # two quiz topics is not counted twice.
        questions_by_topic = {topic_id: [] for topic_id in topic_ids}
        for q in quiz_questions:
            tags = set(q.get("related_topics", [])) | {str(s).split(".")[0] for s in q.get("related_subtopics", [])}
            owner = next((t for t in topic_ids if t in tags), topic_ids[0])
            questions_by_topic[owner].append(q)
        per_topic = {
            topic_id: self.error_analysis_agent.analyze(qs, answers or {}, topic_id=topic_id)
            for topic_id, qs in questions_by_topic.items() if qs
        }
        errors = [error for result in per_topic.values() for error in result["errors"]]
        self._log(
            "Error Analysis Agent", f"{len(errors)} error signal(s) over {len(per_topic)} topic(s)",
            {"topics": topic_ids, "wrong_questions": [q.get("prob_id") for q in wrong_problems]},
            {
                "accuracy_per_topic": {t: r["topic_accuracy"] for t, r in per_topic.items()},
                "questions_per_topic": {t: len(qs) for t, qs in questions_by_topic.items()},
                "errors": errors,
            },
            started,
        )

        student_kb = state.get("student_kb")
        quiz_id = f"{state.get('student_id')}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
        forced = state.get("forced_route") or {}
        retest_of = forced.get("quiz_id") if forced.get("kind") == "retest" else None
        target_subtopics = state.get("quiz_subtopics") or []
        retest_record = None
        # Root cause of the errors, using the configured method (no_kb / kb_low_score /
        # hypothesis). Runs before the outcome is recorded so "recent quizzes" means quizzes
        # before this one. compare() runs every method and weight set on the same quiz.
        started = time.perf_counter()
        weights = resolve_weights(self.root_cause_weights, self.root_cause_preset)
        root_cause_result = self.root_cause_agent.analyze(
            quiz_questions, answers or {}, student_kb,
            method=self.root_cause_method, preset=self.root_cause_preset, weights=self.root_cause_weights,
        )
        comparison = self.root_cause_agent.compare(
            quiz_questions, answers or {}, student_kb, base_preset=self.root_cause_preset,
        )
        # The hypothesis table is always shown, even when another method is selected.
        hypotheses = root_cause_result["hypotheses"] or self.root_cause_agent.analyze(
            quiz_questions, answers or {}, student_kb,
            method="hypothesis", preset=self.root_cause_preset, weights=self.root_cause_weights,
        )["hypotheses"]
        root_cause = root_cause_result["root_cause"]
        compact_comparison = summarize_comparison(comparison)
        if root_cause:
            print("\n🔎 [Root Cause] Hypotheses for this quiz's errors:")
            print(format_hypotheses({"hypotheses": hypotheses}))
            print(f"   => [{METHOD_LABELS[self.root_cause_method]}] {root_cause['summary']}")
        self._log(
            "Root Cause Agent", f"{METHOD_LABELS[self.root_cause_method]}",
            {
                "method": self.root_cause_method,
                "preset": self.root_cause_preset,
                "weights": weights,
                "wrong_answers": [
                    {"prob_id": r["prob_id"], "chosen": r["chosen"], "correct": r["correct"], "subtopics": r["subtopics"]}
                    for r in root_cause_result["responses"] if not r["is_correct"]
                ],
            },
            {"root_cause": root_cause, "hypotheses": hypotheses, "comparison": compact_comparison},
            started,
        )
        if student_kb:
            # One assessment event per quiz topic (its accuracy updates that topic's mastery).
            # The root cause is stored once, on the event of the topic it belongs to.
            responses_by_id = {r["prob_id"]: r for r in root_cause_result["responses"]}
            root_topic = (
                root_cause["topic_id"] if root_cause and root_cause["topic_id"] in per_topic
                else next(iter(per_topic), None)
            )
            mastery_before = dict(student_kb.get("topics", {}))
            for topic_id, analysis in per_topic.items():
                is_root_topic = topic_id == root_topic
                student_kb = record_assessment_outcome(
                    student_kb,
                    topic_id,
                    analysis["topic_accuracy"],
                    [error["signal_id"] for error in analysis["errors"]],
                    {error["signal_id"]: error["description"] for error in analysis["errors"]},
                    question_results=[
                        responses_by_id[q.get("prob_id", q.get("problem_id"))]
                        for q in questions_by_topic[topic_id]
                        if q.get("prob_id", q.get("problem_id")) in responses_by_id
                    ],
                    root_cause=root_cause if is_root_topic else None,
                )
                event = student_kb["pedagogy_history"][-1]
                event["quiz_topics"] = topic_ids
                event["quiz_id"] = quiz_id
                if retest_of:
                    event["retest_of"] = retest_of
                if is_root_topic:
                    # Keep what every method/weight set picked, and the raw parameters of the
                    # candidates, to evaluate methods and refit weights later.
                    event["root_cause_comparison"] = compact_comparison
                    event["hypothesis_candidates"] = comparison.get("candidates", [])
                    event["root_cause_settings"] = {"method": self.root_cause_method,
                                                    "preset": self.root_cause_preset, "weights": weights}
            # Subtopic mastery: every answer updates the subtopics its question tests.
            student_kb, subtopic_changes = update_subtopic_mastery(student_kb, root_cause_result["responses"])
            if retest_of:
                retest_record = self._learning_gain(student_kb, retest_of, quiz_id, target_subtopics,
                                                    root_cause_result["responses"])
                student_kb.setdefault("retests", []).append(retest_record)
            student_kb = remember(student_kb, quiz_entry(
                state.get("user_input"), topic_ids, f"{correct_count}/{total}",
                format_root_cause(root_cause) if root_cause else "",
            ))
            self._save_student_kb(student_kb)
            self._log(
                "Pedagogy Controller", f"Recorded quiz outcome for {', '.join(per_topic)}",
                {"accuracy_per_topic": {t: r["topic_accuracy"] for t, r in per_topic.items()},
                 "root_cause": root_cause and root_cause["subtopic_id"]},
                {"topic_mastery": {
                    t: {"before": mastery_before.get(t), "after": student_kb.get("topics", {}).get(t)}
                    for t in per_topic
                 },
                 "subtopic_mastery": subtopic_changes,
                 "learning_gain": retest_record,
                 "weaknesses": student_kb.get("weaknesses", [])},
            )

        return {
            "student_kb": student_kb,
            "student_answers": answers,
            "graded_details": graded_details,
            "wrong_problems": wrong_problems,
            "score_history": state.get("score_history", []) + [accuracy],
            "round_log": [{
                "type": "quiz_result",
                "score": f"{correct_count}/{total}",
                "accuracy": accuracy,
                "root_cause": root_cause,
            }],
            "response_output": {
                "status": "completed",
                "quiz_score": f"{correct_count}/{total}",
                "accuracy": accuracy,
                "intro_text": intro_text,
                "quiz_topics": topic_ids,
                "accuracy_per_topic": {t: r["topic_accuracy"] for t, r in per_topic.items()},
                "error_signals": errors,
                "root_cause": root_cause,
                "root_cause_method": self.root_cause_method,
                "hypotheses": hypotheses,
                "comparison": comparison,
                "quiz_id": quiz_id,
                "retest": retest_record,
                # For the "why did you get this wrong?" form in the app.
                "wrong_answers": [
                    {"prob_id": r["prob_id"], "chosen": r["chosen"], "correct": r["correct"],
                     "chosen_text": r["chosen_text"], "subtopics": r["subtopics"],
                     "misconception": r.get("misconception"),
                     "question": next((q.get("prob_body") for q in quiz_questions
                                       if q.get("prob_id") == r["prob_id"]), "")}
                    for r in root_cause_result["responses"] if not r["is_correct"]
                ],
            },
            "last_active_agent": "assessment_agent"
        }

    @staticmethod
    def _learning_gain(student_kb: dict, original_quiz_id: str, retest_quiz_id: str,
                       subtopics: list, retest_responses: list) -> dict:
        """Accuracy on the re-tested subtopics before (original quiz) and after (re-test)."""
        def accuracy(responses):
            relevant = [r for r in responses if set(r.get("subtopics", [])) & set(subtopics)]
            return (round(sum(bool(r.get("is_correct")) for r in relevant) / len(relevant), 2)
                    if relevant else None), len(relevant)

        original = [e for e in student_kb.get("pedagogy_history", [])
                    if e.get("event") == "assessment" and e.get("quiz_id") == original_quiz_id
                    and not e.get("retest_of")]
        before, n_before = accuracy([r for e in original for r in e.get("responses", [])])
        after, n_after = accuracy(retest_responses)
        root_event = next((e for e in original if e.get("root_cause")), {})
        remediation_turns = sum(
            e.get("turns", 1) for e in student_kb.get("pedagogy_history", [])
            if e.get("event") == "lesson" and e.get("remediation_for") == original_quiz_id
        )
        return {
            "quiz_id": original_quiz_id,
            "retest_quiz_id": retest_quiz_id,
            "subtopics": subtopics,
            "root_cause_method": (root_event.get("root_cause_settings") or {}).get("method"),
            "root_cause_preset": (root_event.get("root_cause_settings") or {}).get("preset"),
            "accuracy_before": before, "questions_before": n_before,
            "accuracy_after": after, "questions_after": n_after,
            "gain": round(after - before, 2) if before is not None and after is not None else None,
            "remediation_turns": remediation_turns,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _save_trace(self, thread_id: str, trace: list, status: str) -> None:
        if not trace or not TRACE_DIR:
            return
        try:
            os.makedirs(TRACE_DIR, exist_ok=True)
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in thread_id)
            with open(os.path.join(TRACE_DIR, f"{safe_name}.jsonl"), "a", encoding="utf-8") as trace_file:
                trace_file.write(json.dumps({
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "thread_id": thread_id,
                    "status": status,
                    "trace": trace,
                }, default=str) + "\n")
        except OSError as error:            # logging must never break the tutor
            print(f"[Trace] Could not save trace: {error}")

    def _save_student_kb(self, student_kb: dict) -> None:
        import json
        import os

        database = [
            record for record in read_skb_database(self.db_filepath)
            if record.get("student_id") != student_kb.get("student_id")
        ]
        database.append(student_kb)

        parent = os.path.dirname(self.db_filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        temporary_path = self.db_filepath + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as db_file:
            json.dump(database, db_file, indent=2)
        os.replace(temporary_path, self.db_filepath)

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("orchestrator", self.orchestrator_node)
        g.add_node(DIAGNOSIS_NODE, self.diagnosis_node)
        g.add_node(TUTOR_NODE, self.tutor_node)
        g.add_node(ASSESSMENT_NODE, self.assessment_quiz_node)

        g.set_entry_point("orchestrator")

        # Routing out of the orchestrator is done only by its Command(goto=...). There is no
        # conditional edge as well: having both made the tutor run in parallel with every
        # diagnosis and quiz request.
        g.add_edge(DIAGNOSIS_NODE, END)
        g.add_edge(TUTOR_NODE, END)
        g.add_edge(ASSESSMENT_NODE, END)

        return g.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------- public API
    def start(self, student_id: str, user_input: str, student_answers: dict = None,
              chat_history: list = None, thread_id: str = None, forced_route: dict = None) -> dict:
        """Run one turn. `forced_route` skips the routing LLM (app buttons):
        {"kind": "remediation", "quiz_id", "topic_id", "subtopic_id", "subtopic_name", "reason", "misconception"}
        {"kind": "retest", "quiz_id", "topic_id", "subtopics": [...]}"""
        thread_id = thread_id or f"{student_id}-session"
        student_kb = load_student_kb(student_id, self.db_filepath)
        
        # Earlier messages: by default the student's chat memory from the KB (latest 3
        # exchanges, summarised). An explicit chat_history (role/content dicts) overrides it.
        if chat_history is None:
            formatted_messages = history_messages(student_kb)
        else:
            formatted_messages = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in chat_history
            ]
        formatted_messages.append(HumanMessage(content=user_input))

        initial_state = {
            "student_id": student_id,
            "user_input": user_input,
            "chat_history": chat_history or [],
            "messages": formatted_messages,
            "student_answers": student_answers or {},
            "topic_focus": [],
            "orchestrator_instruction": "",
            "orchestrator_reasoning": "",
            "student_kb": student_kb,
            "active_quiz_questions": [],
            "graded_details": [],
            "wrong_problems": [],
            "score_history": [],
            "round_log": [],
            "response_output": None,
            "last_active_agent": None,
            "conversation_mode": "new_request",
            "tutor_question": "",
            "turn_analysis": None,
            "forced_route": forced_route,
            "quiz_subtopics": [],
        }
        return self._invoke(initial_state, thread_id)

    def clear_chat_memory(self, student_id: str) -> None:
        """Forget the stored chat memory (used by "Reset Chat / Session")."""
        student_kb = load_student_kb(student_id, self.db_filepath)
        if student_kb and student_kb.get("chat_memory"):
            student_kb["chat_memory"] = []
            self._save_student_kb(student_kb)

    def resume(self, thread_id: str, answers: dict) -> dict:
        return self._invoke(Command(resume=answers), thread_id)

    def _invoke(self, payload, thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
        self._trace = []
        self._resuming = isinstance(payload, Command)
        result = self.app.invoke(payload, config)
        trace, self._trace = self._trace, []
        self._save_trace(thread_id, trace, "waiting_for_interrupt" if "__interrupt__" in result else "completed")

        if "__interrupt__" in result:
            return {
                "status": "waiting_for_interrupt",
                "thread_id": thread_id,
                "payload": result["__interrupt__"][0].value,
                "trace": trace,
            }
        return {
            "thread_id": thread_id,
            **(result.get("response_output") or {}),
            "round_log": result.get("round_log", []),
            "trace": trace,
        }