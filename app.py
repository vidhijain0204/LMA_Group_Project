# =====================================================================
# STREAMLIT MULTI-AGENT TUTORING SYSTEM (`app.py`)
# =====================================================================
import streamlit as st
import json
from pathlib import Path

from agents.workflow import AdaptiveTutorWorkflow
from agents.orchestrator import load_student_kb, SKB_FILE_PATH
from agents.diagnosis_agent import DiagnosisAgent

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="AI Python Tutoring System",
    page_icon="🎓",
    layout="wide",
)

# Initialize workflow and diagnosis agent.
# Streamlit re-runs this whole script on every click, so the workflow is created once and cached:
# a new workflow would have an empty checkpointer and could not resume a paused diagnosis or quiz.
@st.cache_resource
def get_workflow() -> AdaptiveTutorWorkflow:
    return AdaptiveTutorWorkflow(db_filepath=SKB_FILE_PATH)


workflow = get_workflow()
diag_agent = DiagnosisAgent(db_filepath=SKB_FILE_PATH)

# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================
if "student_id" not in st.session_state:
    st.session_state.student_id = "S001"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "S001-session"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "waiting_for_interrupt" not in st.session_state:
    st.session_state.waiting_for_interrupt = None

# =====================================================================
# SIDEBAR: STUDENT PROFILE & CONTROLS
# =====================================================================
with st.sidebar:
    st.header("🎓 Student Portal")
    
    input_id = st.text_input("Enter Student ID:", value=st.session_state.student_id)
    
    if st.button("Load / Switch Student"):
        st.session_state.student_id = input_id.strip()
        st.session_state.thread_id = f"{input_id.strip()}-session"
        st.session_state.messages = []
        st.session_state.waiting_for_interrupt = None
        st.rerun()

    st.divider()

    kb = load_student_kb(st.session_state.student_id, SKB_FILE_PATH)
    if kb:
        st.success("Status: Registered Student")
        st.markdown(f"**Strengths:** {', '.join(kb.get('strengths', [])) or 'None yet'}")
        st.markdown(f"**Weaknesses:** {', '.join(kb.get('weaknesses', [])) or 'None yet'}")

        pedagogy_history = kb.get("pedagogy_history", [])
        if pedagogy_history:
            with st.expander("Adaptive teaching history"):
                for event in reversed(pedagogy_history[-8:]):
                    topic = event.get("topic_id", "Unknown topic")
                    if event.get("event") == "lesson":
                        outcome = event.get("assessment_accuracy")
                        evidence = (
                            f" | next quiz accuracy: {outcome:.0%}"
                            if outcome is not None
                            else " | awaiting assessment"
                        )
                        st.markdown(
                            f"- **{event.get('strategy', 'unknown').replace('_', ' ').title()}** "
                            f"on {topic}{evidence}"
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
                    st.markdown(
                        f"- Quiz on {entry.get('quiz_topic_id', '?')}: "
                        f"**{entry.get('subtopic_id')} {entry.get('name')}** "
                        f"(P={entry.get('probability', 0):.2f}) - {entry.get('reason', '')}"
                    )

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
                st.rerun()

    st.divider()
    if st.button("Reset Chat / Session"):
        st.session_state.messages = []
        st.session_state.waiting_for_interrupt = None
        st.rerun()

# =====================================================================
# MAIN CHAT & INTERRUPT INTERFACE
# =====================================================================
st.title("🤖 AI Python Tutoring Assistant")
st.markdown(f"Chatting as **{st.session_state.student_id}**. Ask questions, request explanations, or ask for a quiz!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if graph is interrupted (e.g. waiting for diagnosis test or quiz answers)
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
                    st.success("🎉 Knowledge base initialized successfully!")
                    st.rerun()

    elif int_type == "quiz":
        st.warning(f"🎯 **Adaptive Quiz Challenge** — Topic: {interrupt_payload.get('topic_focus')}")
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
                    
                    score_msg = f"📊 **Quiz Completed!** Score: {res.get('quiz_score')} (Accuracy: {res.get('accuracy', 0.0) * 100}%)"
                    error_areas = list(dict.fromkeys(
                        error.get("description", "")
                        for error in res.get("error_signals", [])
                        if error.get("description")
                    ))
                    if error_areas:
                        score_msg += "\n\nAreas to revisit: " + ", ".join(error_areas[:4])
                    hypotheses = res.get("hypotheses") or []
                    root_cause = res.get("root_cause")
                    if root_cause:
                        score_msg += (
                            f"\n\n🔎 **Most likely root cause:** {root_cause['subtopic_id']} "
                            f"{root_cause['name']} (P={root_cause['probability']:.2f}), because "
                            f"{root_cause['reason']}."
                        )
                        score_msg += (
                            "\n\n| Hypothesis | QE | CE | DS | RP | ST | LS | PM' | PW | Score | P |"
                            "\n|---|---|---|---|---|---|---|---|---|---|---|"
                        )
                        for h in hypotheses:
                            p = h["params"]
                            score_msg += (
                                f"\n| {h['subtopic_id']} {h['name']} | {p['QE']:.2f} | {p['CE']:.2f} | "
                                f"{p['DS']:.2f} | {p['RP']:.2f} | {p['ST']:.2f} | {p['LS']:.2f} | "
                                f"{p['PM']:.2f} | {p['PW']:.2f} | {h['score']:+.3f} | {h['probability']:.2f} |"
                            )
                    st.session_state.messages.append({"role": "assistant", "content": score_msg})
                    st.rerun()

else:
    # Standard Chat Input
    if user_prompt := st.chat_input("Ask your tutor a question or request a quiz..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.spinner("Processing request..."):
            result = workflow.start(
                student_id=st.session_state.student_id,
                user_input=user_prompt,
                thread_id=st.session_state.thread_id
            )

            # Check if workflow hit an interrupt (diagnosis or quiz)
            if result.get("status") == "waiting_for_interrupt":
                st.session_state.waiting_for_interrupt = result["payload"]
                st.rerun()
            else:
                # Normal completion (Tutor response or summary)
                response_text = result.get("response_text", "I am here to help you learn Python!")
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()