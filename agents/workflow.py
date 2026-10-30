import operator
from typing import TypedDict, Literal, Any, Annotated, Optional
from agents.llm_factory import get_llm

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, AIMessage

from agents.orchestrator import create_orchestrator_node, load_student_kb, SKB_FILE_PATH
from agents.diagnosis_agent import DiagnosisAgent
from agents.tutor_agent import create_tutor_node, clean_tutor_output
from agents.assessment_agent import create_assessment_node

# --- Configuration & File Path Constants ---
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
    def __init__(self, db_filepath: str = SKB_FILE_PATH):
        self.db_filepath = db_filepath
        self.diagnosis_agent = DiagnosisAgent(db_filepath=db_filepath)

        self.orchestrator_node_fn = create_orchestrator_node()
        self.tutor_node_fn = create_tutor_node(enable_kb_context=True)
        self.assessment_node_fn = create_assessment_node()

        self.checkpointer = MemorySaver()
        self.app = self._build_graph()

    # ------------------------------------------------------------------ nodes
    def orchestrator_node(self, state: AgentState) -> Command:
        print("\n🔵 [Orchestrator] Analyzing student status and intent...")
        return self.orchestrator_node_fn(state)

    def diagnosis_node(self, state: AgentState) -> dict:
        print("\n🟢 [Diagnosis] Administering diagnostic assessment test...")
        answers = interrupt({
            "type": "diagnosis",
            "student_id": state["student_id"],
            "questions": self.diagnosis_agent.get_questions(),
        })
        
        new_kb = self.diagnosis_agent.administer_test(state["student_id"], answers)
        skb_dict = new_kb.model_dump(mode="json")
        
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
        result = self.tutor_node_fn(state)
        response_text = result["messages"][-1].content
        
        return {
            "round_log": [{
                "type": "explanation",
                "focus_topic": state.get("topic_focus"),
                "text": response_text,
            }],
            "response_output": {
                "status": "completed",
                "response_text": response_text,
            },
            "last_active_agent": "tutor_agent"
        }

    def assessment_quiz_node(self, state: AgentState) -> dict:
        print(f"\n🟣 [Assessment] Generating adaptive quiz for topic: {state.get('topic_focus')}")
        result = self.assessment_node_fn(state)
        intro_text = result["messages"][-1].content
        quiz_questions = result["active_quiz_questions"]

        answers = interrupt({
            "type": "quiz",
            "topic_focus": state.get("topic_focus"),
            "intro_text": intro_text,
            "questions": quiz_questions,
        })

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

        return {
            "student_answers": answers,
            "graded_details": graded_details,
            "wrong_problems": wrong_problems,
            "score_history": state.get("score_history", []) + [accuracy],
            "round_log": [{
                "type": "quiz_result",
                "score": f"{correct_count}/{total}",
                "accuracy": accuracy,
            }],
            "response_output": {
                "status": "completed",
                "quiz_score": f"{correct_count}/{total}",
                "accuracy": accuracy,
                "intro_text": intro_text,
            },
            "last_active_agent": "assessment_agent"
        }

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("orchestrator", self.orchestrator_node)
        g.add_node("diagnosis", self.diagnosis_node)
        g.add_node("tutor_agent", self.tutor_node)                  
        g.add_node("assessment_agent", self.assessment_quiz_node)

        g.set_entry_point("orchestrator")

        g.add_edge("diagnosis", END)
        g.add_edge("tutor_agent", END)
        g.add_edge("assessment_agent", END)

        routes = {
            "diagnosis": "diagnosis", 
            "tutor_agent": "tutor_agent", 
            "assessment_quiz": "assessment_agent"
        }
        g.add_conditional_edges("orchestrator", self.route_from_orchestrator, routes)
        
        return g.compile(checkpointer=self.checkpointer)

    def route_from_orchestrator(self, state: AgentState) -> Literal["diagnosis", "tutor_agent", "assessment_quiz"]:
        next_agent = str(state.get("next_agent", "tutor_agent")).lower().strip()
        if "assess" in next_agent or "quiz" in next_agent or "test" in next_agent:
            return "assessment_agent"
        elif "diag" in next_agent:
            return "diagnosis"
        elif "tutor" in next_agent:
            return "tutor_agent"
        return "tutor_agent"

    # ------------------------------------------------------------- public API
    def start(self, student_id: str, user_input: str, student_answers: dict = None,
              chat_history: list = None, thread_id: str = None) -> dict:
        thread_id = thread_id or f"{student_id}-session"
        student_kb = load_student_kb(student_id, self.db_filepath)
        
        formatted_messages = []
        if chat_history:
            for m in chat_history:
                if m["role"] == "user":
                    formatted_messages.append(HumanMessage(content=m["content"]))
                else:
                    formatted_messages.append(AIMessage(content=m["content"]))
        
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
        }
        return self._invoke(initial_state, thread_id)

    def resume(self, thread_id: str, answers: dict) -> dict:
        return self._invoke(Command(resume=answers), thread_id)

    def _invoke(self, payload, thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
        result = self.app.invoke(payload, config)

        if "__interrupt__" in result:
            return {
                "status": "waiting_for_interrupt",
                "thread_id": thread_id,
                "payload": result["__interrupt__"][0].value,
            }
        return {
            "thread_id": thread_id, 
            **(result.get("response_output") or {}),
            "round_log": result.get("round_log", [])
        }