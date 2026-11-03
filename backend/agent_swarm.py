"""Planner -> Analyst -> Transformer -> Critic loop with bounded self-healing retries (LangGraph)."""
from typing import TypedDict
import numpy as np, cv2
from langgraph.graph import StateGraph, END

THRESHOLD, MAX_RETRIES = 0.8, 3

class S(TypedDict):
    img: np.ndarray; goal: str; plan: list; diag: dict; out: np.ndarray; score: float; tries: int; log: list

def planner(s): s["log"].append({"agent": "Planner", "msg": f"Goal '{s['goal']}': analyse, transform, critique"}); s["plan"] = ["analyse", "transform", "critique"]; return s
def analyst(s): s["log"].append({"agent": "Vision Analyst", "msg": f"blur={s['diag']['blur_score']:.0f}"}); return s
def transformer(s):
    k = 1 + s["tries"]; blur = cv2.GaussianBlur(s["img"], (0, 0), 1.2)
    s["out"] = cv2.addWeighted(s["img"], 1 + 0.5 * k, blur, -0.5 * k, 0)  # unsharp mask, stronger on each retry
    s["log"].append({"agent": "Transformer", "msg": f"unsharp strength {0.5 * k:.1f}"}); return s
def critic(s):
    g = cv2.cvtColor(s["out"], cv2.COLOR_RGB2GRAY); v = cv2.Laplacian(g, cv2.CV_64F).var()
    s["score"] = float(min(1.0, v / 300)); s["tries"] += 1
    s["log"].append({"agent": "Critic", "msg": f"score {s['score']:.2f} (threshold {THRESHOLD})"}); return s
def route(s): return END if s["score"] >= THRESHOLD or s["tries"] >= MAX_RETRIES else "transform"

g = StateGraph(S)
for n, f in [("plan", planner), ("analyse", analyst), ("transform", transformer), ("critique", critic)]: g.add_node(n, f)
g.set_entry_point("plan"); g.add_edge("plan", "analyse"); g.add_edge("analyse", "transform"); g.add_edge("transform", "critique")
g.add_conditional_edges("critique", route, {"transform": "transform", END: END})
GRAPH = g.compile()

def run_swarm(img, goal, diag_fn):
    st = GRAPH.invoke({"img": img, "goal": goal, "diag": diag_fn(img), "tries": 0, "log": [], "score": 0.0, "plan": [], "out": img})
    yield from st["log"]; yield {"agent": "Swarm", "msg": f"done, score {st['score']:.2f}, {st['tries']} pass(es)"}
