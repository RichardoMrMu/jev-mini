"""
A small hand-built evaluation set.

Why not import a standard benchmark: the point here is to measure whether
confidence tracks correctness, and that requires items whose difficulty is
deliberately spread out. A set of easy cases produces high accuracy, high
confidence and a flat reliability diagram that proves nothing. So each item
carries a `difficulty` tag and the mix is intentional:

  easy   -- unambiguous, should be near-certain
  medium -- requires reading, one plausible distractor
  hard   -- genuinely ambiguous, where a well-calibrated model SHOULD hedge

The hard cases are the informative ones. Any model can be confident and right
about "URGENT: server down". The question is whether it stays confident when it
shouldn't -- and every `hard` label below marks a case where a human annotator
could reasonably disagree, which is precisely where an honest model ought to
report 0.5 rather than 0.95.

Labels are the author's judgement, written before any model was run against
them. They are not model-generated -- one of the stated weaknesses of the
vendor's own evaluation is that some reference answers came from a strong model
rather than from human annotation.
"""

SUPPORT_TICKETS = [
    # ---- easy ----------------------------------------------------------
    {
        "text": "My credit card was charged twice for order #4471. Please refund the duplicate charge immediately.",
        "category": "billing",
        "urgent": True,
        "difficulty": "easy",
    },
    {
        "text": "Hi, I just wanted to say your new dashboard redesign looks fantastic. Great work!",
        "category": "feedback",
        "urgent": False,
        "difficulty": "easy",
    },
    {
        "text": "The entire production API has been returning 503 for the last 40 minutes. All our customers are affected.",
        "category": "technical",
        "urgent": True,
        "difficulty": "easy",
    },
    {
        "text": "How do I change the email address associated with my account?",
        "category": "account",
        "urgent": False,
        "difficulty": "easy",
    },
    {
        "text": "Please cancel my subscription. I no longer need the service.",
        "category": "account",
        "urgent": False,
        "difficulty": "easy",
    },
    {
        "text": "Your invoice #8823 shows tax applied at 20% but our region should be exempt. Can you re-issue it?",
        "category": "billing",
        "urgent": False,
        "difficulty": "easy",
    },
    # ---- medium --------------------------------------------------------
    {
        "text": "I've been trying to export my data for three days. The button spins and nothing downloads. I have a board meeting Thursday and need this.",
        "category": "technical",
        "urgent": True,
        "difficulty": "medium",
    },
    {
        "text": "We were promised the enterprise tier included SSO. It doesn't appear in our settings. Was this removed?",
        "category": "account",
        "urgent": False,
        "difficulty": "medium",
    },
    {
        "text": "The mobile app crashes when I rotate the screen on the reports page. Not urgent but annoying.",
        "category": "technical",
        "urgent": False,
        "difficulty": "medium",
    },
    {
        "text": "Can you explain why my bill went from $49 to $147 this month? I didn't change anything.",
        "category": "billing",
        "urgent": False,
        "difficulty": "medium",
    },
    {
        "text": "Our team lead left the company and we can't access the admin panel. Nobody else has owner permissions.",
        "category": "account",
        "urgent": True,
        "difficulty": "medium",
    },
    {
        "text": "The search feature returns results from deleted projects. This seems like it might be a privacy issue?",
        "category": "technical",
        "urgent": True,
        "difficulty": "medium",
    },
    # ---- hard: a careful annotator could argue the other way ------------
    {
        "text": "I'd like to understand the pricing for adding 20 seats, and also the seat counter on our current plan looks wrong.",
        # Two genuine categories in one message. 'billing' is the primary ask.
        "category": "billing",
        "urgent": False,
        "difficulty": "hard",
    },
    {
        "text": "Following up on my previous message.",
        # No content whatsoever. A confident answer here is a bad sign.
        "category": "other",
        "urgent": False,
        "difficulty": "hard",
    },
    {
        "text": "This is the third time I'm writing. Still no response. Do you actually read these?",
        # Complaint about service, not about the product. Frustrated but no outage.
        "category": "feedback",
        "urgent": True,
        "difficulty": "hard",
    },
    {
        "text": "Quick question about the API rate limits on the free tier before we commit to a paid plan.",
        # Pre-sales question wearing technical clothing.
        "category": "technical",
        "urgent": False,
        "difficulty": "hard",
    },
    {
        "text": "The payment failed but money left my account. Your system says unpaid. I need this resolved today.",
        # Billing, though it reads like a system fault.
        "category": "billing",
        "urgent": True,
        "difficulty": "hard",
    },
    {
        "text": "Thanks for fixing the login bug. Though now the session expires much faster than before.",
        # Praise plus a new bug report. Primary content is the regression.
        "category": "technical",
        "urgent": False,
        "difficulty": "hard",
    },
]

CATEGORIES = ["billing", "technical", "account", "feedback", "other"]


# Sentiment, with a deliberate concentration of hard cases: sarcasm, mixed
# valence and faint praise are exactly where a confident wrong answer is easy
# to produce and where calibration earns its keep.
REVIEWS = [
    {"text": "Absolutely love it. Best purchase I've made all year.", "sentiment": "positive", "difficulty": "easy"},
    {"text": "Broke after two days. Complete waste of money.", "sentiment": "negative", "difficulty": "easy"},
    {"text": "It arrived on Tuesday in a brown box.", "sentiment": "neutral", "difficulty": "easy"},
    {"text": "Does the job. Nothing special, nothing terrible.", "sentiment": "neutral", "difficulty": "easy"},
    {"text": "The build quality is excellent, but the software is frustrating and crashes often.", "sentiment": "mixed", "difficulty": "medium"},
    {"text": "Good value for the price, though I wish the battery lasted longer.", "sentiment": "positive", "difficulty": "medium"},
    {"text": "I wanted to like this. I really did.", "sentiment": "negative", "difficulty": "hard"},
    {"text": "Well, it certainly is a product that exists.", "sentiment": "negative", "difficulty": "hard"},
    {"text": "Not bad at all, actually.", "sentiment": "positive", "difficulty": "hard"},
    {"text": "It's fine I guess. My expectations were low and they were met.", "sentiment": "negative", "difficulty": "hard"},
    {"text": "Five stars for the customer service. Two stars for everything else.", "sentiment": "mixed", "difficulty": "hard"},
    {"text": "Perfect if you enjoy reading manuals for three hours.", "sentiment": "negative", "difficulty": "hard"},
]

SENTIMENTS = ["positive", "negative", "neutral", "mixed"]


def split_dev_eval(items, dev_ratio=0.35, seed=0):
    """Deterministic, stratified split.

    Temperature must be fitted on data the evaluation never sees, otherwise the
    reported ECE is fitted-on-itself and meaningless. Stratifying by difficulty
    keeps the hard cases from piling into one side.
    """
    import random

    by_diff = {}
    for it in items:
        by_diff.setdefault(it.get("difficulty", "medium"), []).append(it)

    dev, ev = [], []
    rng = random.Random(seed)
    for _, group in sorted(by_diff.items()):
        g = list(group)
        rng.shuffle(g)
        k = max(1, round(len(g) * dev_ratio))
        dev.extend(g[:k])
        ev.extend(g[k:])
    return dev, ev
