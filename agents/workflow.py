import operator
from typing import TypedDict, Literal, Any, Annotated, Optional
from agents.llm_factory import get_llm

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from transformers import AutoModelForCausalLM, AutoTokenizer

from agents.orchestrator import Orchestrator
from agents.diagnosis_agent import DiagnosisAgent
from agents.tutor_agent import TutorAgent

# --- Configuration & File Path Constants ---
DEFAULT_SKB_DB_PATH = "data/skb_database.json"
SLM_MODEL_NAME = "Qwen/Qwen3-4B"
MAX_ITERATIONS = 3
QUIZ_SIZE = 3


class AgentState(TypedDict):
    # inputs
    student_id: str
    user_input: str
    chat_history: list
    student_answers: dict
    # routing
    next_agent: str
    # loop control
    focus_topic: Optional[str]
    iteration: int                 # completed quiz rounds
    max_iterations: int
    # per-round data
    quiz: list
    graded_details: list
    wrong_problems: list
    score_history: list            # accuracy per round
    hypotheses: list               # one record per round (see Error Analysis)
    # outputs
    round_log: Annotated[list, operator.add]   # appended to, never overwritten
    response_output: Any


class AdaptiveTutorWorkflow:
    def __init__(self, db_filepath: str = DEFAULT_SKB_DB_PATH, with_loop: bool = False):
        self.db_filepath = db_filepath
        self.with_loop = with_loop

        self.orchestrator = Orchestrator(db_filepath=db_filepath, llm=get_llm("router"))
        self.diagnosis_agent = DiagnosisAgent(db_filepath=db_filepath)
        self.tutor_agent = TutorAgent(db_filepath=db_filepath, llm=get_llm("tutor"))

        if with_loop:
            from agents.assessment_agent import AssessmentAgent
            from agents.error_analysis_agent import ErrorAnalysisAgent
            from agents.skb_update_agent import SKBUpdateAgent
            self.assessment_agent = AssessmentAgent(db_filepath=db_filepath)
            self.error_analysis_agent = ErrorAnalysisAgent(
                db_filepath=db_filepath, judge_llm=get_llm("judge")
            )
            self.skb_update_agent = SKBUpdateAgent(db_filepath=db_filepath)

        self.checkpointer = MemorySaver()
        self.app = self._build_graph()

    # ------------------------------------------------------------------ nodes
    def orchestrator_node(self, state: AgentState) -> dict:
        print("\n🔵 [Orchestrator] Analyzing student status and intent...")
        student_id = state["student_id"]

        if not self.orchestrator.check_student_exists(student_id):
            print(f"   ↳ NEW student -> [Diagnosis]")
            return {"next_agent": "diagnosis"}

        intent = self.orchestrator.classify_intent(
            state["user_input"], state.get("chat_history", [])
        )
        print(f"   ↳ Intent: {intent}")
        wants_quiz = intent == "ASSESSMENT" and self.with_loop
        return {"next_agent": "assessment_quiz" if wants_quiz else "tutor"}

    def diagnosis_node(self, state: AgentState) -> dict:
        print("\n🟢 [Diagnosis] Asking the diagnostic test...")
        answers = interrupt({
            "type": "diagnosis",
            "student_id": state["student_id"],
            "questions": self.diagnosis_agent.get_questions(),
        })
        skb = self.diagnosis_agent.administer_test(state["student_id"], answers).model_dump(mode="json")
        return {"round_log": [{
            "round": 0,
            "type": "diagnosis",
            "topics": skb["topics"],                 # NEW: per-topic scores for the screen
            "strengths": skb["strengths"],
            "weaknesses": skb["weaknesses"],
            "misconceptions": skb["active_misconceptions"],
        }]}

    def tutor_node(self, state: AgentState) -> dict:
        focus = state.get("focus_topic")
        round_no = state["iteration"] + 1
        print(f"\n🟡 [Tutor] Round {round_no}, focus_topic={focus}")
        explanation = self.tutor_agent.explain(
            student_id=state["student_id"],
            user_input=state["user_input"],
            focus_topic=focus,
            chat_history=state.get("chat_history", []),
        )
        return {"round_log": [{
            "round": round_no,
            "type": "explanation",
            "focus_topic": focus,
            "text": explanation,
        }]}

    def assessment_quiz_node(self, state: AgentState) -> dict:
        print("\n🟣 [Assessment] Selecting quiz...")
        retest_ids = [p["problem_id"] for p in state.get("wrong_problems", [])]
        quiz = self.assessment_agent.get_targeted_quiz(
            student_id=state["student_id"],
            limit=QUIZ_SIZE,
            retest_ids=retest_ids,
        )
        return {"quiz": quiz}

    def assessment_grade_node(self, state: AgentState) -> dict:
        print("\n🟣 [Assessment] Waiting for student answers...")
        explanations = [r for r in state.get("round_log", []) if r["type"] == "explanation"]

        # Graph PAUSES here. Execution resumes with Command(resume=answers).
        answers = interrupt({
            "type": "quiz",
            "round": state["iteration"] + 1,
            "latest_explanation": explanations[-1]["text"] if explanations else None,
            "questions": state["quiz"],
        })

        result = self.assessment_agent.grade_quiz(state["quiz"], answers)
        print(f"   ↳ Score: {result['score']}")
        return {
            "student_answers": answers,
            "graded_details": result["details"],
            "wrong_problems": result["wrong_problems"],
            "iteration": state["iteration"] + 1,
            "score_history": state.get("score_history", []) + [result["accuracy"]],
            "round_log": [{
                "round": state["iteration"] + 1,
                "type": "quiz_result",
                "score": result["score"],
                "accuracy": result["accuracy"],
            }],
        }

    def skb_update_node(self, state: AgentState) -> dict:
        print("\n⚪ [SKB Update] Writing graded results to the KB...")
        self.skb_update_agent.update(state["student_id"], state["graded_details"])
        return {}

    def error_analysis_node(self, state: AgentState) -> dict:
        print("\n🔴 [Error Analysis] Building & testing hypotheses...")
        result = self.error_analysis_agent.analyze(
            student_id=state["student_id"],
            wrong_problems=state["wrong_problems"],
            previous_hypotheses=state.get("hypotheses", []),
            score_history=state["score_history"],
            iteration=state["iteration"],
        )
        return {
            "hypotheses": state.get("hypotheses", []) + [result["round_record"]],
            "focus_topic": result["focus_topic"],
        }

    def finalize_node(self, state: AgentState) -> dict:
        print("\n🏁 [Finalize] Concluding root cause...")
        conclusion = (
            self.error_analysis_agent.conclude(state["hypotheses"])
            if state.get("hypotheses") else None
        )
        return {"response_output": {
            "status": "completed",
            "rounds_completed": state["iteration"],
            "score_history": state["score_history"],
            "hypotheses": state.get("hypotheses", []),
            "root_cause": conclusion,
        }}

    # ---------------------------------------------------------------- routing
    def route_from_orchestrator(
        self, state: AgentState
    ) -> Literal["diagnosis", "tutor", "assessment_quiz"]:
        return state["next_agent"]

    def route_after_update(
        self, state: AgentState
    ) -> Literal["error_analysis", "finalize"]:
        if state["wrong_problems"] and state["iteration"] < state["max_iterations"]:
            return "error_analysis"
        return "finalize"

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("orchestrator", self.orchestrator_node)
        g.add_node("diagnosis", self.diagnosis_node)
        g.add_node("tutor", self.tutor_node)
        g.set_entry_point("orchestrator")

        routes = {"diagnosis": "diagnosis", "tutor": "tutor"}
        g.add_edge("diagnosis", END)        # new student: build KB, then answer the first question

        if self.with_loop:
            g.add_node("assessment_quiz", self.assessment_quiz_node)
            g.add_node("assessment_grade", self.assessment_grade_node)
            g.add_node("skb_update", self.skb_update_node)
            g.add_node("error_analysis", self.error_analysis_node)
            g.add_node("finalize", self.finalize_node)
            routes["assessment_quiz"] = "assessment_quiz"

            g.add_edge("tutor", "assessment_quiz")
            g.add_edge("assessment_quiz", "assessment_grade")
            g.add_edge("assessment_grade", "skb_update")
            g.add_conditional_edges("skb_update", self.route_after_update, {
                "error_analysis": "error_analysis",
                "finalize": "finalize",
            })
            g.add_edge("error_analysis", "tutor")
            g.add_edge("finalize", END)
        else:
            g.add_edge("tutor", END)

        g.add_conditional_edges("orchestrator", self.route_from_orchestrator, routes)
        return g.compile(checkpointer=self.checkpointer)

    def print_graph_ascii(self):
        print("\n--- LangGraph Workflow Architecture Map ---")
        try:
            print(self.app.get_graph().draw_ascii())
        except Exception as e:
            print(f"ASCII generation skipped: {e}")
        print("-------------------------------------------\n")

    # ------------------------------------------------------------- public API
    def start(self, student_id: str, user_input: str, student_answers: dict = None,
              chat_history: list = None, thread_id: str = None) -> dict:
        thread_id = thread_id or f"{student_id}-session"
        initial_state = {
            "student_id": student_id,
            "user_input": user_input,
            "chat_history": chat_history or [],
            "student_answers": student_answers or {},
            "next_agent": "",
            "focus_topic": None,
            "iteration": 0,
            "max_iterations": MAX_ITERATIONS,
            "quiz": [],
            "graded_details": [],
            "wrong_problems": [],
            "score_history": [],
            "hypotheses": [],
            "round_log": [],
            "response_output": None,
        }
        return self._invoke(initial_state, thread_id)

    def resume(self, thread_id: str, answers: dict) -> dict:
        return self._invoke(Command(resume=answers), thread_id)

    def _invoke(self, payload, thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
        result = self.app.invoke(payload, config)

        if "__interrupt__" in result:
            return {
                "status": "waiting_for_answers",
                "thread_id": thread_id,
                "payload": result["__interrupt__"][0].value,
            }
        return {"thread_id": thread_id, **(result.get("response_output") or {}),
                "round_log": result.get("round_log", [])}