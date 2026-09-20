"""
A hand-built evaluation set.

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
shouldn't -- and every `hard` label marks a case where a careful annotator
could reasonably disagree, which is precisely where an honest model ought to
report 0.5 rather than 0.95.

Labels are the author's judgement, written before any model was run against
them. They are not model-generated -- one of the stated weaknesses of the
vendor's own evaluation is that some reference answers came from a strong model
rather than from human annotation.

Sizes: 120 support tickets, 60 reviews. Calibration measured on ~18 items is
dominated by binomial noise: one flipped item moves a 5-bin ECE by several
points, which is why an earlier version of this repo could not tell a real
temperature-scaling effect from a split artefact. These sizes are still small
in absolute terms, but they are enough for the per-difficulty breakdowns to
mean something and for a fitted temperature to be more signal than noise.

ANNOTATION CONVENTIONS (applied uniformly, so the hard cases stay defensible):
  category -- the department that must ACT. A message that sounds technical but
              whose resolution is a refund is billing.
  urgent   -- would a reasonable ops team page someone today? Revenue actively
              blocked, data loss, security exposure, or a hard external
              deadline. Annoyance, however loudly expressed, is not urgent.
  severity -- 1 trivial, 3 real problem with a workaround, 5 total loss of
              service or money.
"""

# ---------------------------------------------------------------------------
# Support tickets: 120 items (40 easy / 40 medium / 40 hard)
# ---------------------------------------------------------------------------

SUPPORT_TICKETS = [
    # ================= EASY (40) ==========================================
    {"text": "My credit card was charged twice for order #4471. Please refund the duplicate charge immediately.", "category": "billing", "urgent": True, "difficulty": "easy"},
    {"text": "Hi, I just wanted to say your new dashboard redesign looks fantastic. Great work!", "category": "feedback", "urgent": False, "difficulty": "easy"},
    {"text": "The entire production API has been returning 503 for the last 40 minutes. All our customers are affected.", "category": "technical", "urgent": True, "difficulty": "easy"},
    {"text": "How do I change the email address associated with my account?", "category": "account", "urgent": False, "difficulty": "easy"},
    {"text": "Please cancel my subscription. I no longer need the service.", "category": "account", "urgent": False, "difficulty": "easy"},
    {"text": "Your invoice #8823 shows tax applied at 20% but our region should be exempt. Can you re-issue it?", "category": "billing", "urgent": False, "difficulty": "easy"},
    {"text": "Where can I download a PDF copy of last month's invoice?", "category": "billing", "urgent": False, "difficulty": "easy"},
    {"text": "The export button on the reports page throws a 500 error every single time.", "category": "technical", "urgent": False, "difficulty": "easy"},
    {"text": "I forgot my password and the reset email never arrives. Can you help me get back in?", "category": "account", "urgent": False, "difficulty": "easy"},
    {"text": "Just wanted to drop a note that your support team handled my last issue brilliantly. Thank you.", "category": "feedback", "urgent": False, "difficulty": "easy"},
    {"text": "We need to add three more users to our plan. What is the process?", "category": "account", "urgent": False, "difficulty": "easy"},
    {"text": "Our database integration has been down since the 3am deploy. Nothing is syncing and orders are piling up.", "category": "technical", "urgent": True, "difficulty": "easy"},
    {"text": "You billed me $499 but my plan is $99. Please correct this and refund the difference today.", "category": "billing", "urgent": True, "difficulty": "easy"},
    {"text": "Could you point me to the documentation for the webhooks API?", "category": "technical", "urgent": False, "difficulty": "easy"},
    {"text": "I love the dark mode you shipped last week. Small thing, big difference.", "category": "feedback", "urgent": False, "difficulty": "easy"},
    {"text": "Please update the billing address on our account to 40 Rue de Rivoli, Paris.", "category": "billing", "urgent": False, "difficulty": "easy"},
    {"text": "All our API keys stopped working simultaneously about ten minutes ago. Production is down.", "category": "technical", "urgent": True, "difficulty": "easy"},
    {"text": "How do I enable two-factor authentication on my login?", "category": "account", "urgent": False, "difficulty": "easy"},
    {"text": "The charge on my statement says PENDING but the money is already gone from my balance.", "category": "billing", "urgent": False, "difficulty": "easy"},
    {"text": "Your onboarding flow is the smoothest I've used in this category. Nice job.", "category": "feedback", "urgent": False, "difficulty": "easy"},
    {"text": "I would like to upgrade from the Starter plan to the Pro plan.", "category": "account", "urgent": False, "difficulty": "easy"},
    {"text": "The mobile app shows a blank white screen on launch since this morning's update.", "category": "technical", "urgent": True, "difficulty": "easy"},
    {"text": "Can you tell me when our annual renewal date is?", "category": "billing", "urgent": False, "difficulty": "easy"},
    {"text": "Please delete my account and all associated data permanently.", "category": "account", "urgent": False, "difficulty": "easy"},
    {"text": "The CSV import silently drops every row that contains a comma inside a quoted field.", "category": "technical", "urgent": False, "difficulty": "easy"},
    {"text": "Fantastic release notes this month. Clear, honest, and actually readable.", "category": "feedback", "urgent": False, "difficulty": "easy"},
    {"text": "We were double-billed for both January and February. Two duplicate charges total.", "category": "billing", "urgent": True, "difficulty": "easy"},
    {"text": "How do I transfer ownership of a workspace to a different user?", "category": "account", "urgent": False, "difficulty": "easy"},
    {"text": "Search returns zero results for every query since yesterday evening.", "category": "technical", "urgent": True, "difficulty": "easy"},
    {"text": "Your pricing page is much clearer than it was last year. Appreciated.", "category": "feedback", "urgent": False, "difficulty": "easy"},
    {"text": "I need a VAT invoice for the last three months for our accounting department.", "category": "billing", "urgent": False, "difficulty": "easy"},
    {"text": "Can I change my username without losing my history?", "category": "account", "urgent": False, "difficulty": "easy"},
    {"text": "Uploads over 10MB fail with a timeout every time, smaller ones work fine.", "category": "technical", "urgent": False, "difficulty": "easy"},
    {"text": "Subject: hello. Body: hello.", "category": "other", "urgent": False, "difficulty": "easy"},
    {"text": "The refund you promised two weeks ago still has not arrived in my account.", "category": "billing", "urgent": True, "difficulty": "easy"},
    {"text": "Please remove the former employee jane@acme.com from our team seats.", "category": "account", "urgent": False, "difficulty": "easy"},
    {"text": "Every webhook we receive has a malformed JSON body since the v3 rollout.", "category": "technical", "urgent": True, "difficulty": "easy"},
    {"text": "Really glad you finally added keyboard shortcuts. Huge quality of life win.", "category": "feedback", "urgent": False, "difficulty": "easy"},
    {"text": "What credit cards do you accept for payment?", "category": "billing", "urgent": False, "difficulty": "easy"},
    {"text": "I cannot log in at all. It says my account is suspended but I have no idea why.", "category": "account", "urgent": True, "difficulty": "easy"},

    # ================= MEDIUM (40) ========================================
    {"text": "I've been trying to export my data for three days. The button spins and nothing downloads. I have a board meeting Thursday and need this.", "category": "technical", "urgent": True, "difficulty": "medium"},
    {"text": "We were promised the enterprise tier included SSO. It doesn't appear in our settings. Was this removed?", "category": "account", "urgent": False, "difficulty": "medium"},
    {"text": "The mobile app crashes when I rotate the screen on the reports page. Not urgent but annoying.", "category": "technical", "urgent": False, "difficulty": "medium"},
    {"text": "Can you explain why my bill went from $49 to $147 this month? I didn't change anything.", "category": "billing", "urgent": False, "difficulty": "medium"},
    {"text": "Our team lead left the company and we can't access the admin panel. Nobody else has owner permissions.", "category": "account", "urgent": True, "difficulty": "medium"},
    {"text": "The search feature returns results from deleted projects. This seems like it might be a privacy issue?", "category": "technical", "urgent": True, "difficulty": "medium"},
    {"text": "Our monthly usage report shows 40,000 API calls but our own logs say 12,000. Which number are we billed on?", "category": "billing", "urgent": False, "difficulty": "medium"},
    {"text": "After the SSO migration half our team can log in and half get redirected in a loop.", "category": "technical", "urgent": True, "difficulty": "medium"},
    {"text": "We would like to move from monthly to annual billing mid-cycle. How is the proration handled?", "category": "billing", "urgent": False, "difficulty": "medium"},
    {"text": "Notifications arrive roughly six hours late, which makes the alerting feature useless for us.", "category": "technical", "urgent": False, "difficulty": "medium"},
    {"text": "Someone on our team deleted a project by accident. Is there any way to restore it?", "category": "technical", "urgent": True, "difficulty": "medium"},
    {"text": "The seat count on our invoice says 25 but we only ever activated 18 users.", "category": "billing", "urgent": False, "difficulty": "medium"},
    {"text": "I set my timezone to CET but all timestamps still render in UTC.", "category": "technical", "urgent": False, "difficulty": "medium"},
    {"text": "Our compliance team needs to know where our data is physically stored before we renew.", "category": "account", "urgent": False, "difficulty": "medium"},
    {"text": "The trial was supposed to last 30 days but we got locked out on day 14.", "category": "account", "urgent": True, "difficulty": "medium"},
    {"text": "Rate limiting kicks in at around 40 requests per second even though our plan says 100.", "category": "technical", "urgent": True, "difficulty": "medium"},
    {"text": "Could you clarify whether the Pro plan's 'unlimited projects' has a soft cap? We are at 300.", "category": "account", "urgent": False, "difficulty": "medium"},
    {"text": "Two of our invoices have the same number but different amounts. Our auditor flagged it.", "category": "billing", "urgent": False, "difficulty": "medium"},
    {"text": "Dashboard charts render fine on Chrome but are completely blank on Safari.", "category": "technical", "urgent": False, "difficulty": "medium"},
    {"text": "We need to add a purchase order number to all future invoices for procurement.", "category": "billing", "urgent": False, "difficulty": "medium"},
    {"text": "The audit log stops recording after about 1000 entries and silently overwrites the oldest.", "category": "technical", "urgent": True, "difficulty": "medium"},
    {"text": "Is there a way to restrict team members to read-only access on specific projects?", "category": "account", "urgent": False, "difficulty": "medium"},
    {"text": "Our scheduled reports stopped arriving after we changed the workspace name.", "category": "technical", "urgent": False, "difficulty": "medium"},
    {"text": "The discount code from your conference booth was rejected at checkout yesterday.", "category": "billing", "urgent": False, "difficulty": "medium"},
    {"text": "We keep getting logged out every 15 minutes since the security update. It is making work impossible.", "category": "technical", "urgent": True, "difficulty": "medium"},
    {"text": "Can we get a sandbox environment separate from production for our integration tests?", "category": "account", "urgent": False, "difficulty": "medium"},
    {"text": "The API docs show a 'metadata' field but the actual response never includes it.", "category": "technical", "urgent": False, "difficulty": "medium"},
    {"text": "Our card expired and the automatic retry appears to have suspended the account rather than notifying us.", "category": "billing", "urgent": True, "difficulty": "medium"},
    {"text": "Bulk delete removes the items from the list but they reappear after a refresh.", "category": "technical", "urgent": False, "difficulty": "medium"},
    {"text": "We are evaluating whether to renew. Could someone walk us through what changed in v4?", "category": "account", "urgent": False, "difficulty": "medium"},
    {"text": "Emails from your system land in spam for everyone at our domain.", "category": "technical", "urgent": False, "difficulty": "medium"},
    {"text": "The usage graph and the billing page disagree by about 15% every month.", "category": "billing", "urgent": False, "difficulty": "medium"},
    {"text": "Our integration broke after you deprecated the v2 endpoint without any notice we received.", "category": "technical", "urgent": True, "difficulty": "medium"},
    {"text": "How do we set up a shared team inbox rather than individual notifications?", "category": "account", "urgent": False, "difficulty": "medium"},
    {"text": "Autosave sometimes loses the last few edits when the connection drops briefly.", "category": "technical", "urgent": True, "difficulty": "medium"},
    {"text": "We paid the annual plan upfront but the account still shows as monthly.", "category": "billing", "urgent": True, "difficulty": "medium"},
    {"text": "Is the uptime figure on your status page measured per-region or globally?", "category": "technical", "urgent": False, "difficulty": "medium"},
    {"text": "The onboarding checklist keeps reappearing for users who already completed it.", "category": "technical", "urgent": False, "difficulty": "medium"},
    {"text": "Please confirm whether cancelling now still gives us access until the period ends.", "category": "account", "urgent": False, "difficulty": "medium"},
    {"text": "Our admin invited 12 people but only 7 received the invitation email.", "category": "technical", "urgent": False, "difficulty": "medium"},

    # ================= HARD (40) ==========================================
    # Each of these has a defensible alternative reading. That is the point.
    {"text": "I'd like to understand the pricing for adding 20 seats, and also the seat counter on our current plan looks wrong.", "category": "billing", "urgent": False, "difficulty": "hard"},
    {"text": "Following up on my previous message.", "category": "other", "urgent": False, "difficulty": "hard"},
    {"text": "This is the third time I'm writing. Still no response. Do you actually read these?", "category": "feedback", "urgent": True, "difficulty": "hard"},
    {"text": "Quick question about the API rate limits on the free tier before we commit to a paid plan.", "category": "technical", "urgent": False, "difficulty": "hard"},
    {"text": "The payment failed but money left my account. Your system says unpaid. I need this resolved today.", "category": "billing", "urgent": True, "difficulty": "hard"},
    {"text": "Thanks for fixing the login bug. Though now the session expires much faster than before.", "category": "technical", "urgent": False, "difficulty": "hard"},
    {"text": "Per my last email.", "category": "other", "urgent": False, "difficulty": "hard"},
    {"text": "I'm not sure if this is a bug or if I'm holding it wrong, but the totals never match what I expect.", "category": "technical", "urgent": False, "difficulty": "hard"},
    {"text": "We're being charged for a feature we were told was included. Either the sales rep was wrong or the billing is.", "category": "billing", "urgent": False, "difficulty": "hard"},
    {"text": "Honestly at this point I'd just like to know whether this product is still maintained.", "category": "feedback", "urgent": False, "difficulty": "hard"},
    {"text": "Can someone from engineering call me? I've explained this to three support agents already.", "category": "other", "urgent": True, "difficulty": "hard"},
    {"text": "The export works, technically. It just takes 40 minutes and times out if you look at it funny.", "category": "technical", "urgent": False, "difficulty": "hard"},
    {"text": "Is it normal for the invoice to arrive before the service period starts, or is that a bug?", "category": "billing", "urgent": False, "difficulty": "hard"},
    {"text": "Our security team has questions about the data retention clause before we can expand the contract.", "category": "account", "urgent": False, "difficulty": "hard"},
    {"text": "It says my trial expired but I signed up yesterday. Either the clock is wrong or I misread something.", "category": "account", "urgent": True, "difficulty": "hard"},
    {"text": "Not really a complaint, just an observation: the new nav has more clicks to get anywhere.", "category": "feedback", "urgent": False, "difficulty": "hard"},
    {"text": "We may have exposed an API key in a public repo. What should we do?", "category": "technical", "urgent": True, "difficulty": "hard"},
    {"text": "Do you have a student discount, and if so does it apply to the team plan or only individuals?", "category": "billing", "urgent": False, "difficulty": "hard"},
    {"text": "Every time I think I've found the setting, it turns out to be somewhere else. Is there a search for settings?", "category": "technical", "urgent": False, "difficulty": "hard"},
    {"text": "Our renewal is in four days and procurement still hasn't received the quote they asked for last month.", "category": "billing", "urgent": True, "difficulty": "hard"},
    {"text": "Ignore my last three tickets, I figured it out. Leaving this here in case someone else hits it.", "category": "other", "urgent": False, "difficulty": "hard"},
    {"text": "The docs say one thing, the API does another, and the support article says a third. Which is right?", "category": "technical", "urgent": False, "difficulty": "hard"},
    {"text": "Would you consider a feature where the report could be scheduled? Or does that already exist and I missed it?", "category": "feedback", "urgent": False, "difficulty": "hard"},
    {"text": "We got an email saying our account will be deleted in 7 days. We are a paying customer. Is this a phishing attempt or yours?", "category": "account", "urgent": True, "difficulty": "hard"},
    {"text": "I was told by chat support that a refund was issued. My bank says nothing was received. Someone is mistaken.", "category": "billing", "urgent": True, "difficulty": "hard"},
    {"text": "The performance is fine for us but our users in Asia report it being unusable.", "category": "technical", "urgent": True, "difficulty": "hard"},
    {"text": "I don't need help, I just want to register that this migration has cost us two weeks.", "category": "feedback", "urgent": False, "difficulty": "hard"},
    {"text": "Are the numbers on the dashboard live or cached? Our finance team is reconciling against them.", "category": "technical", "urgent": False, "difficulty": "hard"},
    {"text": "We'd like to downgrade, but only if we keep the custom fields. If not, we'll stay on the current plan.", "category": "account", "urgent": False, "difficulty": "hard"},
    {"text": "There's a typo on your checkout page that says 'Anual'. Minor, but it's on the page where people pay you.", "category": "feedback", "urgent": False, "difficulty": "hard"},
    {"text": "Something is wrong. I can't be more specific than that, it just doesn't work like it did last week.", "category": "technical", "urgent": False, "difficulty": "hard"},
    {"text": "Our invoice shows a credit we don't recognise. We'd rather flag it than quietly benefit from an error.", "category": "billing", "urgent": False, "difficulty": "hard"},
    {"text": "Is there a reason the API returns 200 with an error object inside instead of a 4xx status?", "category": "technical", "urgent": False, "difficulty": "hard"},
    {"text": "My colleague says she reported this in March. I can't find the ticket. Starting over, apparently.", "category": "other", "urgent": False, "difficulty": "hard"},
    {"text": "We're happy overall. The only thing keeping us from recommending you is the billing portal.", "category": "feedback", "urgent": False, "difficulty": "hard"},
    {"text": "If I delete a user, does that delete their data too? I need to know before I click it.", "category": "account", "urgent": True, "difficulty": "hard"},
    {"text": "The charge went through twice but one shows as reversed. Do I need to do anything or will it settle?", "category": "billing", "urgent": False, "difficulty": "hard"},
    {"text": "Your status page says all systems operational. From here, they are not.", "category": "technical", "urgent": True, "difficulty": "hard"},
    {"text": "I'd rather not open a ticket for every small thing. Is there a better channel for papercuts?", "category": "feedback", "urgent": False, "difficulty": "hard"},
    {"text": "We need this fixed before Monday or we'll have to explain to our board why we chose you.", "category": "other", "urgent": True, "difficulty": "hard"},
]

CATEGORIES = ["billing", "technical", "account", "feedback", "other"]


# ---------------------------------------------------------------------------
# Reviews: 60 items, with a deliberate concentration of hard cases.
# Sarcasm, mixed valence and faint praise are exactly where a confident wrong
# answer is easy to produce and where calibration earns its keep.
# ---------------------------------------------------------------------------

REVIEWS = [
    # ---- easy (18) ----
    {"text": "Absolutely love it. Best purchase I've made all year.", "sentiment": "positive", "difficulty": "easy"},
    {"text": "Broke after two days. Complete waste of money.", "sentiment": "negative", "difficulty": "easy"},
    {"text": "It arrived on Tuesday in a brown box.", "sentiment": "neutral", "difficulty": "easy"},
    {"text": "Does the job. Nothing special, nothing terrible.", "sentiment": "neutral", "difficulty": "easy"},
    {"text": "Outstanding quality and it shipped two days early. Delighted.", "sentiment": "positive", "difficulty": "easy"},
    {"text": "Arrived damaged, the seller ignored three emails. Avoid.", "sentiment": "negative", "difficulty": "easy"},
    {"text": "The package measures 30x20x10 cm and weighs about a kilo.", "sentiment": "neutral", "difficulty": "easy"},
    {"text": "Exceeded every expectation. I have already ordered a second one.", "sentiment": "positive", "difficulty": "easy"},
    {"text": "Stopped charging within a week. Returned it.", "sentiment": "negative", "difficulty": "easy"},
    {"text": "Comes in three colours according to the listing.", "sentiment": "neutral", "difficulty": "easy"},
    {"text": "Genuinely the best in its class. Worth every penny.", "sentiment": "positive", "difficulty": "easy"},
    {"text": "Terrible. Loud, flimsy, and it smells of solvent.", "sentiment": "negative", "difficulty": "easy"},
    {"text": "Standard packaging, standard delivery, standard product.", "sentiment": "neutral", "difficulty": "easy"},
    {"text": "I have recommended this to four colleagues already. Superb.", "sentiment": "positive", "difficulty": "easy"},
    {"text": "Do not buy. Mine failed on day three and support never replied.", "sentiment": "negative", "difficulty": "easy"},
    {"text": "It is a black rectangular device with two buttons.", "sentiment": "neutral", "difficulty": "easy"},
    {"text": "Perfect fit, beautiful finish, arrived on time. No complaints at all.", "sentiment": "positive", "difficulty": "easy"},
    {"text": "Cheaply made and overpriced. Deeply disappointed.", "sentiment": "negative", "difficulty": "easy"},

    # ---- medium (18) ----
    {"text": "The build quality is excellent, but the software is frustrating and crashes often.", "sentiment": "mixed", "difficulty": "medium"},
    {"text": "Good value for the price, though I wish the battery lasted longer.", "sentiment": "positive", "difficulty": "medium"},
    {"text": "Great sound, terrible battery. Depends what you care about.", "sentiment": "mixed", "difficulty": "medium"},
    {"text": "Solid product held back by awful documentation.", "sentiment": "mixed", "difficulty": "medium"},
    {"text": "Works well enough once you get past the setup, which took an hour.", "sentiment": "positive", "difficulty": "medium"},
    {"text": "Beautiful design. Shame about the noise it makes under load.", "sentiment": "mixed", "difficulty": "medium"},
    {"text": "I like it more than the previous model, though that's a low bar.", "sentiment": "positive", "difficulty": "medium"},
    {"text": "Fast and reliable, but the subscription pricing is hard to justify.", "sentiment": "mixed", "difficulty": "medium"},
    {"text": "No real complaints. Also no real enthusiasm.", "sentiment": "neutral", "difficulty": "medium"},
    {"text": "The hardware is superb and the app is a disaster. Buy it and use third-party software.", "sentiment": "mixed", "difficulty": "medium"},
    {"text": "Decent for the money, I'd buy again if it were slightly quieter.", "sentiment": "positive", "difficulty": "medium"},
    {"text": "Does what it says. The packaging was excessive and wasteful though.", "sentiment": "mixed", "difficulty": "medium"},
    {"text": "Better than I expected given the reviews, but still not great.", "sentiment": "neutral", "difficulty": "medium"},
    {"text": "Excellent customer service rescued what was otherwise a poor experience.", "sentiment": "mixed", "difficulty": "medium"},
    {"text": "Comfortable and light. The strap broke after a month, so mixed feelings.", "sentiment": "mixed", "difficulty": "medium"},
    {"text": "Pretty good overall, minor gripes about the finish.", "sentiment": "positive", "difficulty": "medium"},
    {"text": "It works. I wanted to love it and I merely tolerate it.", "sentiment": "neutral", "difficulty": "medium"},
    {"text": "Powerful but the learning curve nearly made me return it.", "sentiment": "mixed", "difficulty": "medium"},

    # ---- hard (24): sarcasm, litotes, faint praise, damning by structure ----
    {"text": "I wanted to like this. I really did.", "sentiment": "negative", "difficulty": "hard"},
    {"text": "Well, it certainly is a product that exists.", "sentiment": "negative", "difficulty": "hard"},
    {"text": "Not bad at all, actually.", "sentiment": "positive", "difficulty": "hard"},
    {"text": "It's fine I guess. My expectations were low and they were met.", "sentiment": "negative", "difficulty": "hard"},
    {"text": "Five stars for the customer service. Two stars for everything else.", "sentiment": "mixed", "difficulty": "hard"},
    {"text": "Perfect if you enjoy reading manuals for three hours.", "sentiment": "negative", "difficulty": "hard"},
    {"text": "Ten out of ten would be generous. Nine, then.", "sentiment": "positive", "difficulty": "hard"},
    {"text": "I cannot fault it, which is not quite the same as loving it.", "sentiment": "positive", "difficulty": "hard"},
    {"text": "Brilliant, assuming you never need the thing it was sold to do.", "sentiment": "negative", "difficulty": "hard"},
    {"text": "It's not the worst thing I've bought this month.", "sentiment": "negative", "difficulty": "hard"},
    {"text": "Bold of them to ship this and call it finished.", "sentiment": "negative", "difficulty": "hard"},
    {"text": "Surprisingly competent for the price bracket.", "sentiment": "positive", "difficulty": "hard"},
    {"text": "I've owned worse. I've also owned better. Mostly better.", "sentiment": "negative", "difficulty": "hard"},
    {"text": "If mediocrity had a flagship, this would be it.", "sentiment": "negative", "difficulty": "hard"},
    {"text": "Hard to dislike, hard to recommend.", "sentiment": "neutral", "difficulty": "hard"},
    {"text": "Nothing here is bad, exactly.", "sentiment": "neutral", "difficulty": "hard"},
    {"text": "The one star is for the box. The box was nice.", "sentiment": "negative", "difficulty": "hard"},
    {"text": "It does everything it promises, which turns out to be less than I assumed.", "sentiment": "negative", "difficulty": "hard"},
    {"text": "Against my better judgement, I'm keeping it.", "sentiment": "positive", "difficulty": "hard"},
    {"text": "My only complaint is that I didn't buy it sooner.", "sentiment": "positive", "difficulty": "hard"},
    {"text": "Arrived late, packaged badly, and works flawlessly.", "sentiment": "mixed", "difficulty": "hard"},
    {"text": "You get exactly what you pay for, and I paid very little.", "sentiment": "neutral", "difficulty": "hard"},
    {"text": "Somehow both over-engineered and under-tested.", "sentiment": "negative", "difficulty": "hard"},
    {"text": "I'd call it a bargain if it had lasted another month.", "sentiment": "negative", "difficulty": "hard"},
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
