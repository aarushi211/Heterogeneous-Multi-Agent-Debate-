"""
seeds.py — Seed question bank for H-MAD Scenario 2: Code-Switching

10 causal/logical reasoning questions drawn from XCOPA-style prompts.
Each question is chosen to:
  1. Have a clear, defensible correct answer (so the Judge can evaluate fairly)
  2. Support genuine pushback in Turn 3 (the Opponent can challenge a nuance)
  3. Cover diverse domains (physics, biology, logic, everyday causality)

Structure:
  SEEDS — list of dicts, each with:
    "id"       : unique identifier
    "domain"   : topic category
    "question" : the seed question passed to the Opponent
    "answer_hint" : the correct answer (not shown to any agent — for post-hoc evaluation only)
    "pushback_angle" : the natural follow-up/challenge angle the Opponent should explore in Turn 3
"""

SEEDS: list[dict] = [
    {
        "id": "S01",
        "domain": "physics",
        "question": (
            "If you drop a heavy ball and a light ball from the same height at the same time, "
            "which one hits the ground first?"
        ),
        "answer_hint": (
            "Both hit at the same time in a vacuum (Galileo). In air, the heavier one "
            "hits marginally faster due to lower drag-to-mass ratio, but the difference "
            "is negligible for dense objects."
        ),
        "pushback_angle": (
            "Challenge whether air resistance makes a difference — "
            "push the Proponent to be precise about the vacuum vs. real-world distinction."
        ),
    },
    {
        "id": "S02",
        "domain": "biology",
        "question": (
            "If a plant is kept in complete darkness for several weeks, what happens to it, "
            "and why?"
        ),
        "answer_hint": (
            "The plant undergoes etiolation: it grows tall and spindly, loses chlorophyll "
            "(turns yellow/white), and eventually dies because it cannot perform photosynthesis "
            "to produce glucose."
        ),
        "pushback_angle": (
            "Ask whether the plant could survive on stored energy or soil nutrients alone "
            "without any light."
        ),
    },
    {
        "id": "S03",
        "domain": "chemistry",
        "question": (
            "If you mix baking soda and vinegar together, what happens, "
            "and what is the chemical explanation?"
        ),
        "answer_hint": (
            "An acid-base reaction: NaHCO3 + CH3COOH → CO2 (gas bubbles) + water + sodium acetate. "
            "The fizzing is CO2 escaping."
        ),
        "pushback_angle": (
            "Push the Proponent on whether the reaction produces any heat (it is endothermic — "
            "the mixture cools slightly) to test depth of understanding."
        ),
    },
    {
        "id": "S04",
        "domain": "earth science",
        "question": (
            "During a thunderstorm, why do we always see lightning before we hear thunder, "
            "even when the strike is nearby?"
        ),
        "answer_hint": (
            "Light travels at ~3×10^8 m/s; sound travels at ~343 m/s in air. "
            "The difference in propagation speed means light arrives almost instantly "
            "while sound takes ~3 seconds per kilometer of distance."
        ),
        "pushback_angle": (
            "Ask what happens if the lightning strike is extremely close — "
            "is there any scenario where you hear thunder simultaneously with the flash?"
        ),
    },
    {
        "id": "S05",
        "domain": "thermodynamics",
        "question": (
            "If you leave a glass of ice water on a table in a warm, humid room, "
            "water droplets form on the outside of the glass. Where does that water come from?"
        ),
        "answer_hint": (
            "Condensation: the cold glass surface cools the surrounding air below its dew point, "
            "causing water vapour in the air to condense into liquid droplets on the glass. "
            "The water comes from the air, not from inside the glass."
        ),
        "pushback_angle": (
            "Challenge whether the water could be seeping through the glass material itself, "
            "and demand the Proponent rule it out."
        ),
    },
    {
        "id": "S06",
        "domain": "astronomy",
        "question": (
            "Why does the Moon appear to change shape — from a thin crescent to a full circle "
            "and back — over the course of a month?"
        ),
        "answer_hint": (
            "The Moon does not change shape; we see different fractions of its sunlit half "
            "depending on its orbital position relative to Earth and the Sun. "
            "The cycle takes ~29.5 days (synodic month)."
        ),
        "pushback_angle": (
            "Ask whether Earth's shadow causes the phases (it does not — "
            "that is a lunar eclipse, which is different). Test whether the Proponent "
            "correctly distinguishes phases from eclipses."
        ),
    },
    {
        "id": "S07",
        "domain": "mechanics",
        "question": (
            "If you spin a raw egg and a hard-boiled egg on a table, "
            "how can you tell which is which without cracking them open?"
        ),
        "answer_hint": (
            "The hard-boiled egg spins smoothly and steadily; the raw egg wobbles because "
            "its liquid interior sloshes and lags behind the shell (rotational inertia of "
            "a non-rigid body). Briefly stopping and releasing both: the raw egg restarts "
            "because the liquid inside continues rotating."
        ),
        "pushback_angle": (
            "Push on the stop-and-release test — ask the Proponent to explain "
            "precisely why the raw egg restarts in terms of angular momentum."
        ),
    },
    {
        "id": "S08",
        "domain": "cognitive science",
        "question": (
            "If a person goes without sleep for 72 hours straight, "
            "what specific effects does this have on their cognitive performance?"
        ),
        "answer_hint": (
            "After 72 hours: severe impairment in working memory, attention, and executive function; "
            "micro-sleep episodes; hallucinations (visual and auditory); reaction times comparable "
            "to being legally drunk (~0.1% BAC); and significant emotional dysregulation."
        ),
        "pushback_angle": (
            "Challenge whether willpower or caffeine can fully compensate for these effects, "
            "and push for evidence-based claims."
        ),
    },
    {
        "id": "S09",
        "domain": "logic / probability",
        "question": (
            "You flip a fair coin 10 times and get heads every single time. "
            "What is the probability of getting heads on the 11th flip, and why?"
        ),
        "answer_hint": (
            "Exactly 0.5 (50%). Each flip is an independent event; prior outcomes do not "
            "affect future flips for a fair coin. The Gambler's Fallacy would incorrectly "
            "predict a higher probability of tails."
        ),
        "pushback_angle": (
            "Invoke the Gambler's Fallacy as a challenge — argue that tails is 'overdue' — "
            "and force the Proponent to explain independence of events clearly."
        ),
    },
    {
        "id": "S10",
        "domain": "everyday physics",
        "question": (
            "If you are in a moving elevator that suddenly has its cable cut, "
            "would jumping just before it hits the ground save you, and why or why not?"
        ),
        "answer_hint": (
            "No, it would not save you in any meaningful way. In free fall the elevator and "
            "you fall at the same rate (weightlessness). You cannot jump relative to the "
            "elevator floor because there is no normal force. Even if you could, the relative "
            "speed reduction from a human jump (~2 m/s) is negligible against a terminal "
            "velocity impact (~55+ m/s for a tall building)."
        ),
        "pushback_angle": (
            "Push back that a jump does technically reduce relative impact velocity by some amount — "
            "force the Proponent to quantify why that reduction is negligible."
        ),
    },
]


def get_seed(seed_id: str) -> dict:
    """Returns a single seed dict by ID (e.g., 'S01'). Raises KeyError if not found."""
    for s in SEEDS:
        if s["id"] == seed_id:
            return s
    raise KeyError(f"Seed ID '{seed_id}' not found. Available: {[s['id'] for s in SEEDS]}")


def list_seeds() -> None:
    """Prints a summary table of all seeds."""
    print(f"{'ID':<5} {'Domain':<20} {'Question (truncated)'}")
    print("-" * 70)
    for s in SEEDS:
        print(f"{s['id']:<5} {s['domain']:<20} {s['question'][:45]}...")


if __name__ == "__main__":
    list_seeds()
