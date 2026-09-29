# What we did, explained simply

## The game we're playing

People with Parkinson's go to the doctor many times over the years. Each time, the doctor gives them a **"how shaky/stiff are you" score**. Higher = worse.

But the doctor's score is **messy**:

- 💊 **The medicine hides the symptoms.** If the patient took their pill an hour ago, they look better than they really are. It's like measuring how tired someone is right after they drank a coffee.
- 🙅 **The "no medicine" test is often skipped** because it's unpleasant (the patient has to stop their pills). So 4 times out of 10, that score is just missing.
- 🎲 **Doctors don't all score the same way.**

The organisers worked out the **real, clean score** for every visit. Our job: **guess that clean score using only the messy stuff.**

We're graded on **how far off our guesses are, on average** (in score points). Smaller = better.

---

## The one big idea

Imagine a kid's **height** measured at every doctor visit, but the ruler is wobbly. Each measurement is a bit wrong.

- If you look at **one** measurement, you're stuck with its error.
- If you look at **all** of them together, you can draw a nice smooth growth curve through them, and the curve is much closer to the truth than any single measurement.

That's exactly what's going on here. A patient's real score is a **smooth line that slowly goes up over the years** (the disease slowly gets worse, it never gets better). Every visit is a **wobbly measurement** of that line.

The official guide looked at **each visit alone**. We looked at **all of a patient's visits together**. That's why we did so much better.

---

## How we checked our guesses were honest

The competition tests us on **patients we've never seen**.

So when we practised, we always **hid whole patients**: we learned from 80% of the patients and tested on the other 20%, like studying with one set of exam questions and being tested on different ones. If we had mixed the same patient into both, it'd be like seeing the exam answers in advance: great practice score, bad real score.

We did this the same way for **every** attempt, so we could compare them fairly.

---

## The staircase: what we did, step by step

The number is **how far off our guesses were on Kaggle** (lower = better).

| Step | What we did, in kid words | How far off |
|---|---|---|
| 1 | Guess the **same number for everyone** (the average). Dumb on purpose, just to have something to beat. | 16.4 |
| 2 | A simple formula: "more years sick + worse measured score → worse real score". Plus a trick: we **tell it when a measurement was missing**, because "the doctor skipped the test" is a clue too. | 8.4 |
| 3 | A smarter model: lots of little **"if this, then that" questions** stacked together (decision trees). It can handle missing values by itself. | 7.2 |
| 4–5 | Gave it the **gene** and **group** columns. Didn't help: those don't really change how fast people get worse. (This is where the official guide stops.) | 7.1 |
| 6 | 💡 **The big one.** For each visit, we also told the model about **the patient's other visits**: their average score, which direction it's going, etc. The wobbly-ruler trick. | **3.8** |
| 7 | **Undo the medicine effect.** We learned "1 hour after the pill, the score looks about half as bad as it really is; 3 hours after, a bit less hidden…" and fixed every measurement before using it. | 3.7 |
| 8 | **Connect the dots smoothly.** The real score is a smooth curve, so we smoothed each patient's guesses into a gentle curve instead of zig-zags. | 3.6 |
| 9 | Let the patient's line **bend a little** (real disease curves aren't perfectly straight), and tell the model **how trustworthy** each patient's measurements are. | 3.5 |
| 10 | **Turned the knobs** on the model (how fast it learns, how careful it is…) by trying 30 combinations and keeping the best. | 3.4 |
| 11 | **Asked 4 models and averaged their answers**, like asking several friends and taking the middle answer. | 3.35 |
| 12 | 💡 **Everyone reacts to medicine differently.** For some people the pill works really well, for others less. When a patient did both tests (with and without medicine) at some visits, we measured **their own** reaction and used it to fix their other visits. | 3.14 |
| 13 | Averaged 4 models again, on top of step 12. Barely better, and 4× slower. | 3.13 |
| 14 | Told the model **how far each visit is from a real "no medicine" measurement**, and made sure each patient's curve **never goes down** (the disease doesn't get better). | **3.12** |

**From 16.4 → 3.12.** The official guide's best was 7.1, so we're off by **less than half** as much.

---

## Why we picked the step-14 model

- 🏆 **It's the best** on our own practice tests *and* on Kaggle's real test.
- ✅ **Our practice scores always matched Kaggle's scores**, so we trust it's not a fluke.
- 🧠 **Every piece has a reason** that comes from the data or from how the disease and the medicine work: smooth disease curve, medicine wearing off over a few hours, people reacting differently to medicine.
- 🧼 **It's one model**, not a pile of models, so it's simpler to explain, and it still beats the piles from steps 11 and 13.

---

## Things we tried that didn't work (and that's fine)

Showing what *didn't* work proves we tested things properly.

- Changing the "alpha" knob on the simple formula → no difference.
- Using gene and group → no difference.
- Smoothing with a **straight** line instead of a curve → worse (the real curve bends).
- A fancy statistics method to guess each patient's curve → no better than what we had.
- Fixing the no-medicine score as a *percentage* instead of *"minus a few points"* → worse.

---

## How we worked (the "methodology" in one breath)

1. **Look at the data first** and understand it (that's how we found the smooth-curve idea).
2. **Change one thing at a time.**
3. **Test it fairly** (hide whole patients).
4. **Keep it only if it helps.**
5. **Look at where we're still wrong**, figure out why, and fix that next.
6. **Write everything down**: every attempt has its script, its explanation, its report on Skore Hub, and its Kaggle submission.

---

## If someone asks you tricky questions

**"Isn't looking at other visits of the same patient cheating?"**
No. We only use the messy measurements, never the answers, and Kaggle gives us the same kind of visit history for the test patients too.

**"How do you know it's not just lucky?"**
We always tested on patients the model had never seen, and the Kaggle scores (totally unseen patients) matched ours every single time.

**"What was the most important thing you did?"**
Looking at the data before building models. That's how we noticed each patient follows a smooth curve, and using that cut the error in half.
