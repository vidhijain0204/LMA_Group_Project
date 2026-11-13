"""Streamlit user interface of the AI Python tutoring system.

Run with:  streamlit run app.py

Where it fits:
    This is the only file the student sees. It sends every chat message to
    AdaptiveTutorWorkflow.start() (agents/workflow.py), which routes it through the
    orchestrator to the diagnosis, tutor or assessment agent. When the workflow pauses
    for a diagnostic test or a quiz, this file shows the questions as a form and
    passes the answers back with workflow.resume().

What the page contains:
    - Sidebar: choose/switch student (anonymous ids like S101), root-cause settings
      (method, A/B condition by student id, custom weights for the hypothesis score
      Score(H) built from LS/CE/RP/PW/PM), and a view of the student's stored
      knowledge base (SKB, data/skb_database.json): strengths, weaknesses, teaching
      history, root causes, topic scores.
    - Main area: the chat, the diagnosis/quiz forms, the quiz result with its root
      cause, buttons to review the root cause or re-test it, a form where the student
      labels why they got each question wrong (saved through agents/evaluation_store.py
      as ground truth for evaluation), and an "Agent flow" panel showing each agent's
      input and output.

Note: Streamlit re-runs this whole script after every click, so anything that must
survive between runs is kept in st.session_state.
"""
import streamlit as st
import json
from pathlib import Path

from agents.workflow import AdaptiveTutorWorkflow
from agents.orchestrator import load_student_kb, SKB_FILE_PATH
from agents.diagnosis_agent import DiagnosisAgent
from agents.struggle_tracker import load_catalog
from agents.pedagogy_controller import compact_pedagogy_history
from agents.root_cause_agent import (
    DEFAULT_PRESET, METHOD_LABELS, METHODS, PARAM_LABELS, PARAMS, WEIGHT_PRESETS, SIGNS,
)
from agents import evaluation_store
from agents.orchestrator import read_skb_database

# ---------------------------------------------------------------- page setup
st.set_page_config(
    page_title="AI Python Tutoring System",
    page_icon="🎓",
    layout="wide",
)

# Streamlit re-runs this whole script on every click, so the workflow is created once and cached:
# a new workflow would have an empty checkpointer and could not resume a paused diagnosis or quiz.
@st.cache_resource
def get_workflow() -> AdaptiveTutorWorkflow:
    """Create the workflow once per server process (cached by Streamlit)."""
    # Build the topic classifier now so the first student message is not slowed down by it.
    from agents.topic_classifier import get_classifier
    get_classifier()
    return AdaptiveTutorWorkflow(db_filepath=SKB_FILE_PATH)


workflow = get_workflow()
diag_agent = DiagnosisAgent(db_filepath=SKB_FILE_PATH)

# ---------------------------------------------------------------- session state
# Default values on the first run. waiting_for_interrupt holds the paused diagnosis/quiz
# payload (questions to show), or None when the chat is free.
if "student_id" not in st.session_state:
    st.session_state.student_id = "S001"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "S001-session"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "waiting_for_interrupt" not in st.session_state:
    st.session_state.waiting_for_interrupt = None
# Agent steps of a start() call that paused for a diagnosis/quiz; shown with the final answer.
if "pending_trace" not in st.session_state:
    st.session_state.pending_trace = []

# Id -> readable name lookups from the topic catalog.
TOPIC_NAMES = load_catalog()["topic_names"]
SUBTOPIC_NAMES = load_catalog()["subtopic_names"]
MISCONCEPTIONS = load_catalog()["misconceptions"]


def topic_label(topic_ids) -> str:
    """Turn one topic id or a list of ids into "T1 Name, T2 Name" text."""
    topic_ids = topic_ids if isinstance(topic_ids, list) else [topic_ids]
    return ", ".join(f"{t} {TOPIC_NAMES.get(t, '')}".strip() for t in topic_ids)


# ---------------------------------------------------------------- display helpers
AGENT_ICONS = {
    "Orchestrator": "🧭",
    "Diagnosis Agent": "🩺",
    "Pedagogy Controller": "🧠",
    "Tutor Agent": "👩‍🏫",
    "Assessment Agent": "🎯",
    "Error Analysis Agent": "🧪",
    "Root Cause Agent": "🔎",
}


def render_trace(trace: list) -> None:
    """Collapsible timeline of the agents that ran, with the input and output of each step."""
    if not trace:
        return
    total_ms = sum(step.get("duration_ms") or 0 for step in trace)
    # Order of agents that ran, with repeated consecutive steps of the same agent merged.
    path = []
    for step in trace:
        if not path or path[-1] != step["agent"]:
            path.append(step["agent"])
    with st.expander(f"🛰️ Agent flow · {len(trace)} steps · {total_ms / 1000:.1f}s"):
        st.caption(" → ".join(f"{AGENT_ICONS.get(a, '⚙️')} {a}" for a in path))
        for index, step in enumerate(trace, 1):
            with st.container(border=True):
                duration = f" · `{step['duration_ms']} ms`" if step.get("duration_ms") is not None else ""
                st.markdown(
                    f"**{index}. {AGENT_ICONS.get(step['agent'], '⚙️')} {step['agent']}** — "
                    f"{step['title']}{duration}"
                )
                tab_in, tab_out = st.tabs(["📥 Input", "📤 Output"])
                with tab_in:
                    st.json(step.get("input") or {}, expanded=1)
                with tab_out:
                    st.json(step.get("output") or {}, expanded=1)


def render_root_cause_details(details: dict) -> None:
    """Show the hypothesis table and what each method / weight set picked, for evaluation.

    details holds "root_cause", "hypotheses" and "comparison" from the quiz result.
    """
    if not details:
        return
    with st.expander("📐 Root-cause evidence & method comparison"):
        hypotheses = details.get("hypotheses") or []
        if hypotheses:
            st.markdown("**Hypotheses (top 3 candidate subtopics)**")
            st.dataframe(
                [
                    {
                        "Hypothesis": f"{h['subtopic_id']} {h['name']}",
                        **{("PM'" if k == "PM" else k): h["params"][k] for k in PARAMS},
                        "Score": h["score"],
                        "P(H)": h["probability"],
                    }
                    for h in hypotheses
                ],
                hide_index=True, width="stretch",
            )
        comparison = details.get("comparison")
        if comparison:
            primary = (details.get("root_cause") or {}).get("subtopic_id")
            # One row per method / preset / ablation; votes counts how many picked each subtopic.
            rows, votes = [], {}
            for group, label in (("methods", "Method"), ("presets", "Weights"), ("ablations", "Ablation")):
                for name, entry in comparison[group].items():
                    root = entry if group == "methods" else entry.get("root_cause")
                    topic = f"{root['topic_id']} {root.get('topic_name') or ''}".strip() if root else "—"
                    subtopic = f"{root['subtopic_id']} {root['name']}" if root else "—"
                    rows.append({
                        "Type": label,
                        "Name": METHOD_LABELS.get(name, name),
                        "Selected topic": topic,
                        "Selected subtopic": subtopic,
                        "Same as shown reason": bool(root) and root["subtopic_id"] == primary,
                    })
                    if root:
                        votes.setdefault((topic, subtopic), []).append(name)
            st.markdown("**What each method / weight combination picks on this quiz**")
            for (topic, subtopic), names in sorted(votes.items(), key=lambda item: -len(item[1])):
                st.markdown(f"- **{topic} → {subtopic}** — picked by {len(names)} of {len(rows)}")
            st.dataframe(rows, hide_index=True, width="stretch")


def handle_result(result: dict) -> None:
    """Show a workflow result: pause for a diagnosis/quiz, or add the tutor's reply."""
    if result.get("status") == "waiting_for_interrupt":
        st.session_state.waiting_for_interrupt = result["payload"]
        st.session_state.pending_trace = result.get("trace", [])
        return
    response_text = result.get("response_text", "I am here to help you learn Python!")
    if result.get("conversation_mode") == "follow_up":
        response_text = (
            f"_↩️ Continuing the lesson ({result.get('pedagogy_strategy', '').replace('_', ' ')}) "
            f"— your reply was taken as an answer to my question._\n\n{response_text}"
        )
    elif result.get("conversation_mode") == "remediation":
        response_text = f"_📘 Review lesson on the root cause of your last quiz._\n\n{response_text}"
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "trace": result.get("trace", []),
    })


def run_turn(display_text: str, user_input: str, forced_route: dict = None) -> None:
    """Send one message to the workflow and redraw the page.

    display_text is what the chat shows; user_input is what the agents receive.
    forced_route is set by the review / re-test buttons to skip the orchestrator's choice.
    """
    st.session_state.messages.append({"role": "user", "content": display_text})
    with st.spinner("Processing request..."):
        # Earlier messages come from the student's chat memory in the KB
        # (latest 3 exchanges, stored as summaries) — see agents/chat_memory.py.
        handle_result(workflow.start(
            student_id=st.session_state.student_id,
            user_input=user_input,
            thread_id=st.session_state.thread_id,
            forced_route=forced_route,
        ))
    st.rerun()


def quiz_result_message(res: dict) -> dict:
    """Build the chat message for a graded quiz.

    The text has the score, accuracy per topic, the learning gain for a re-test, and
    the root cause. The extra keys let render_message() show the evidence table,
    the review/re-test buttons and the agent trace.
    """
    score_msg = (
        f"📊 **Quiz Completed!** Score: {res.get('quiz_score')} "
        f"(Accuracy: {res.get('accuracy', 0.0):.0%})"
    )
    per_topic = res.get("accuracy_per_topic") or {}
    if per_topic:
        score_msg += "\n\n" + " · ".join(
            f"{topic_label(t)}: **{'-' if acc is None else f'{acc:.0%}'}**"
            for t, acc in per_topic.items()
        )
    retest = res.get("retest")
    if retest:
        before, after, gain = retest.get("accuracy_before"), retest.get("accuracy_after"), retest.get("gain")
        fmt = lambda v: "-" if v is None else f"{v:.0%}"
        score_msg += (
            f"\n\n📈 **Re-test on {', '.join(retest['subtopics'])}:** before {fmt(before)} → "
            f"after {fmt(after)}" + (f" (gain **{gain:+.0%}**)" if gain is not None else "")
        )
    root_cause = res.get("root_cause")
    if root_cause:
        # One line: topic -> subtopic -> reason (-> likely misconception).
        score_msg += (
            f"\n\n🔎 **Actual reason for the MCQ errors** "
            f"_({METHOD_LABELS[res.get('root_cause_method', 'hypothesis')]})_\n\n"
            f"> **{root_cause['topic_id']} {root_cause.get('topic_name', '')}** → "
            f"**{root_cause['subtopic_id']} {root_cause['name']}** → {root_cause['reason']}"
        )
        if root_cause.get("misconception"):
            score_msg += f"\n>\n> 💡 Likely misconception: _{root_cause['misconception']['description']}_"
    elif res.get("accuracy") == 1.0:
        score_msg += "\n\n✅ No wrong answers — nothing to diagnose."
    return {
        "role": "assistant",
        "content": score_msg,
        "root_cause_details": root_cause and {
            "root_cause": root_cause,
            "hypotheses": res.get("hypotheses"),
            "comparison": res.get("comparison"),
        },
        "quiz": {
            "quiz_id": res.get("quiz_id"),
            "root_cause": root_cause,
            "wrong_answers": res.get("wrong_answers") or [],
            "is_retest": bool(retest),
        },
        "trace": st.session_state.pending_trace + res.get("trace", []),
    }


def render_label_form(quiz: dict) -> None:
    """Ask the student why they got each question wrong, and save it as an evaluation label.

    These answers are the "ground truth" used to check how well the root-cause methods work.
    The form is shown only once per quiz.
    """
    quiz_id = quiz.get("quiz_id")
    if quiz_id in evaluation_store.labelled_quiz_ids(st.session_state.student_id):
        st.caption("🧾 Thanks — your answers about this quiz were saved.")
        return
    with st.expander("🧾 Help us evaluate: why did you get these wrong? (1 minute)"):
        with st.form(f"labels_{quiz_id}"):
            per_question, subtopic_choices = [], {}
            for i, wrong in enumerate(quiz["wrong_answers"], 1):
                st.markdown(f"**{i}. {wrong.get('question', '')}**  \nYou chose _{wrong.get('chosen_text') or wrong.get('chosen')}_")
                # Radio label -> (reason kind, subtopic id or misconception id).
                options = {f"I did not understand: {sid} {SUBTOPIC_NAMES.get(sid, '')}": ("subtopic", sid)
                           for sid in wrong["subtopics"]}
                for sid in wrong["subtopics"]:
                    subtopic_choices[sid] = f"{sid} {SUBTOPIC_NAMES.get(sid, '')}"
                if wrong.get("misconception"):
                    desc = MISCONCEPTIONS.get(wrong["misconception"], wrong["misconception"])
                    options[f"I believed something wrong: {desc}"] = ("misconception", wrong["misconception"])
                options["I guessed / did not know at all"] = ("guess", None)
                options["Careless mistake / I misread the question"] = ("careless", None)
                choice = st.radio("Main reason", list(options), key=f"lab_{quiz_id}_{wrong['prob_id']}", index=None)
                kind, value = options.get(choice, (None, None))
                per_question.append({"prob_id": wrong["prob_id"], "reason": kind,
                                     "subtopic_id": value if kind == "subtopic" else None,
                                     "misconception": value if kind == "misconception" else None})
            overall = st.selectbox("Which ONE of these caused most of your mistakes?",
                                   ["Not sure"] + list(subtopic_choices.values()), key=f"lab_overall_{quiz_id}")
            if st.form_submit_button("Save my answers"):
                true_subtopic = next((sid for sid, label in subtopic_choices.items() if label == overall), None)
                evaluation_store.append("labels", {
                    "student_id": st.session_state.student_id, "quiz_id": quiz_id, "source": "student",
                    "true_subtopic": true_subtopic, "per_question": per_question,
                })
                st.rerun()


def render_quiz_actions(quiz: dict) -> None:
    """Close the loop after a quiz: review the root cause, then re-test it."""
    root = quiz.get("root_cause")
    if not root:
        return
    if quiz.get("wrong_answers"):
        render_label_form(quiz)
    sid, name = root["subtopic_id"], root["name"]
    left, right = st.columns(2)
    if left.button(f"📘 Review {sid} {name}", key=f"review_{quiz['quiz_id']}", width="stretch"):
        run_turn(f"📘 Review the root cause: {sid} {name}", f"Help me understand {sid} {name}.", {
            "kind": "remediation", "quiz_id": quiz["quiz_id"], "topic_id": root["topic_id"],
            "subtopic_id": sid, "subtopic_name": name, "reason": root["reason"],
            "misconception": (root.get("misconception") or {}).get("description"),
        })
    if right.button(f"🔁 Re-test me on {sid}", key=f"retest_{quiz['quiz_id']}", width="stretch",
                    help="A short quiz on this subtopic only, to measure what the review changed."):
        run_turn(f"🔁 Re-test me on {sid} {name}", f"Re-test me on {sid} {name}.", {
            "kind": "retest", "quiz_id": quiz["quiz_id"], "topic_id": root["topic_id"], "subtopics": [sid],
        })


def render_message(message: dict, is_latest_quiz: bool = False) -> None:
    """Draw one chat message with its extras (root-cause details, quiz buttons, trace)."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_root_cause_details(message.get("root_cause_details"))
        if is_latest_quiz and message.get("quiz") and not st.session_state.waiting_for_interrupt:
            render_quiz_actions(message["quiz"])
        render_trace(message.get("trace"))

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🎓 Student Portal")

    input_id = st.text_input("Enter Student ID:", value=st.session_state.student_id)

    def switch_student(student_id: str) -> None:
        """Start a fresh chat session for another student id."""
        st.session_state.student_id = student_id
        st.session_state.thread_id = f"{student_id}-session"
        st.session_state.messages = []
        st.session_state.waiting_for_interrupt = None
        st.session_state.pending_trace = []
        st.rerun()

    if st.button("Load / Switch Student"):
        switch_student(input_id.strip())
    if st.button("🆕 New anonymous student", help="Next free id (S101, S102, ...). Never use real names."):
        switch_student(evaluation_store.next_anonymous_id(
            r.get("student_id", "") for r in read_skb_database(SKB_FILE_PATH)))

    st.divider()

    # Root-cause settings: which method decides the stored root cause, and with which weights.
    with st.expander("🔎 Root-cause settings"):
        auto_condition = st.toggle(
            "Assign method by student id (A/B study)", value=False,
            help="Each student gets one method (by id number mod 3), so the learning gain after "
                 "remediation can be compared between methods.",
        )
        digits = "".join(ch for ch in st.session_state.student_id if ch.isdigit())
        if auto_condition and digits:
            method = METHODS[int(digits) % len(METHODS)]
            st.info(f"Condition for {st.session_state.student_id}: **{METHOD_LABELS[method]}**")
        else:
            method = st.selectbox(
                "Method", METHODS, index=METHODS.index("hypothesis"),
                format_func=lambda m: METHOD_LABELS[m],
            )
        # Off: every parameter has the same weight (plain average). On: sliders, starting from
        # the weights learned by scripts/fit_weights.py if they exist, else from equal weights.
        custom = st.toggle("Custom weights", value=False)
        weights = dict(WEIGHT_PRESETS[DEFAULT_PRESET])
        if custom:
            weights = dict(WEIGHT_PRESETS.get("learned", weights))
            for key in PARAMS:
                weights[key] = st.slider(
                    f"{'−' if SIGNS[key] < 0 else '+'} {PARAM_LABELS[key]} ({key})",
                    0.0, 1.0, float(weights[key]), 0.05, key=f"w_{key}",
                )
        st.caption(
            "Score(H) = " + " ".join(
                f"{'−' if SIGNS[k] < 0 else '+'} {weights[k]:.2f}·{'PM′' if k == 'PM' else k}" for k in PARAMS
            ).lstrip("+ ")
        )
        workflow.root_cause_method = method
        workflow.root_cause_preset = DEFAULT_PRESET
        workflow.root_cause_weights = weights if custom else None

    st.divider()

    kb = load_student_kb(st.session_state.student_id, SKB_FILE_PATH)
    if kb:
        st.success("Status: Registered Student")
        st.markdown(f"**Strengths:** {', '.join(kb.get('strengths', [])) or 'None yet'}")
        st.markdown(f"**Weaknesses:** {', '.join(kb.get('weaknesses', [])) or 'None yet'}")

        pedagogy_history = kb.get("pedagogy_history", [])
        if pedagogy_history:
            with st.expander("Adaptive teaching history"):
                st.caption("Lesson = teaching block on a topic with one strategy; "
                           "the score is from the first quiz after it.")
                for event in reversed(compact_pedagogy_history(pedagogy_history)[-8:]):
                    topic = event.get("topic_id", "Unknown topic")
                    if event.get("event") == "lesson":
                        turns = event.get("turns", 1)
                        outcome = event.get("assessment_accuracy")
                        if "assessment_accuracy" not in event:
                            evidence = " | not quizzed yet"
                        elif outcome is None:
                            evidence = " | quiz not topic-aligned"
                        else:
                            evidence = f" | next quiz accuracy: {outcome:.0%}"
                        st.markdown(
                            f"- **{event.get('strategy', 'unknown').replace('_', ' ').title()}** "
                            f"on {topic} · {turns} message{'s' if turns != 1 else ''}{evidence}"
                        )
                    elif event.get("event") == "assessment":
                        accuracy = event.get("accuracy")
                        result = f"{accuracy:.0%}" if accuracy is not None else "not topic-aligned"
                        st.markdown(
                            f"- Quiz on {topic}: **{result}**; "
                            f"{len(event.get('error_signals', []))} error signal(s)"
                        )

        root_causes = kb.get("root_cause_history", [])
        if root_causes:
            with st.expander("Root causes of quiz errors"):
                for entry in reversed(root_causes[-5:]):
                    summary = entry.get("summary") or (
                        f"{entry.get('subtopic_id')} {entry.get('name')} → {entry.get('reason', '')}"
                    )
                    st.markdown(f"- Quiz on {entry.get('quiz_topic_id', '?')}: {summary}")

        with st.expander("🗂️ Full knowledge base (stored values)"):
            st.caption(f"Saved in {SKB_FILE_PATH} · last update {kb.get('latest_update_time', '-')}")
            st.json(kb, expanded=1)

        with st.expander("View Topic Scores"):
            for tid, score in kb.get("topics", {}).items():
                st.progress(score, text=f"{tid}: {score}")
    else:
        st.warning("Status: New Student / Unregistered")
        st.info("Type a message or click below to trigger the diagnostic onboarding test.")
        if st.button("Start Diagnostic Test", disabled=bool(st.session_state.waiting_for_interrupt)):
            # The orchestrator sends every new student to the diagnosis, whatever the message says.
            result = workflow.start(
                student_id=st.session_state.student_id,
                user_input="I am a new student. Please start my diagnostic test.",
                thread_id=st.session_state.thread_id,
            )
            if result.get("status") == "waiting_for_interrupt":
                st.session_state.waiting_for_interrupt = result["payload"]
                st.session_state.pending_trace = result.get("trace", [])
                st.rerun()

    st.divider()
    if st.button("Reset Chat / Session"):
        workflow.clear_chat_memory(st.session_state.student_id)
        st.session_state.messages = []
        st.session_state.waiting_for_interrupt = None
        st.session_state.pending_trace = []
        st.rerun()

# ---------------------------------------------------------------- main chat area
st.title("🤖 AI Python Tutoring Assistant")
st.markdown(f"Chatting as **{st.session_state.student_id}**. Ask questions, request explanations, or ask for a quiz!")

# Display chat history (the buttons and form after a quiz only on the latest quiz result)
latest_quiz = max((i for i, m in enumerate(st.session_state.messages) if m.get("quiz")), default=None)
for index, message in enumerate(st.session_state.messages):
    render_message(message, is_latest_quiz=index == latest_quiz)

# If the workflow is paused, show the diagnosis or quiz form instead of the chat input.
interrupt_payload = st.session_state.waiting_for_interrupt

if interrupt_payload:
    int_type = interrupt_payload.get("type")

    if int_type == "diagnosis":
        st.warning("📝 **Diagnostic Assessment Required** (New Student Onboarding)")
        questions = interrupt_payload.get("questions", [])

        with st.form("interrupt_diagnosis_form"):
            submitted_answers = {}
            for idx, q in enumerate(questions, 1):
                st.subheader(f"Question {idx}: {q['body']}")
                options = q.get("options", q.get("option", []))
                choice = st.radio(
                    f"Select answer for Q{idx}:",
                    options=options,
                    key=f"diag_q_{q.get('problem_id', idx)}"
                )
                submitted_answers[q.get('problem_id')] = choice
                st.divider()

            submit_diag = st.form_submit_button("Submit Diagnostic Answers")
            if submit_diag:
                with st.spinner("Processing diagnostic test and building Knowledge Base..."):
                    res = workflow.resume(st.session_state.thread_id, submitted_answers)
                    st.session_state.waiting_for_interrupt = None
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "🎉 **Diagnostic completed** — your knowledge base is ready. "
                                   "Ask a question or request a quiz!",
                        "trace": st.session_state.pending_trace + res.get("trace", []),
                    })
                    st.session_state.pending_trace = []
                    st.rerun()

    elif int_type == "quiz":
        st.warning(f"🎯 **Adaptive Quiz Challenge** — Topics: {topic_label(interrupt_payload.get('topic_focus') or [])}")
        st.markdown(interrupt_payload.get("intro_text", ""))

        questions = interrupt_payload.get("questions", [])
        with st.form("interrupt_quiz_form"):
            quiz_answers = {}
            for idx, q in enumerate(questions, 1):
                st.subheader(f"Quiz Question {idx}: {q.get('prob_body')}")
                options = q.get("options", {})
                # options is a dict like {"A": "...", "B": "..."}
                option_labels = list(options.keys())
                option_texts = [f"{k}: {v}" for k, v in options.items()]

                choice_text = st.radio(
                    f"Select answer for Q{idx}:",
                    options=option_texts,
                    key=f"quiz_q_{q.get('prob_id')}"
                )
                # Extract chosen key (e.g., 'A' from 'A: The Python interpreter')
                chosen_key = choice_text.split(":")[0].strip() if choice_text else None
                quiz_answers[q.get('prob_id')] = chosen_key
                st.divider()

            submit_quiz = st.form_submit_button("Submit Quiz Answers")
            if submit_quiz:
                with st.spinner("Grading quiz and updating progress..."):
                    res = workflow.resume(st.session_state.thread_id, quiz_answers)
                    st.session_state.waiting_for_interrupt = None
                    st.session_state.messages.append(quiz_result_message(res))
                    st.session_state.pending_trace = []
                    st.rerun()

else:
    # Normal chat: send the typed message to the workflow.
    if user_prompt := st.chat_input("Ask your tutor a question or request a quiz..."):
        run_turn(user_prompt, user_prompt)
