"""Updated-facts benchmark: does NoMem notice when a newer memory changes an older one?

No LLM anywhere. Each **update** case plants an older statement, ~40 turns of
unrelated chatter, then a newer statement that changes it, more chatter,
and asks a question about it. Each **control** case plants two statements
that look alike but do *not* contradict each other (a second pet, a
different person, a past event, an elaboration). We measure:

- **detection recall**   update cases where a conflict links old -> new
- **false-flag rate**    control cases where a conflict wrongly links them
- **both surfaced**      update cases whose budgeted context shows both
                         memories (each with its date) and a conflict marker
- **LoCoMo flag rate**   conflicts raised per 1,000 real conversational turns,
                         with a sample to eyeball (there is no gold label)

Usage:
    python benchmarks/updates_bench.py [--verbose] [--embedder model2vec]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nomem import ApproxCounter, NoMem  # noqa: E402

# (older, newer, question, needle in older, needle in newer)
UPDATES = [
    ("I live in Pune.", "I moved to Bangalore last week!", "where do I live?", "Pune", "Bangalore"),
    ("My address is 12 Baker Street.", "My new address is 5 Elm Road.", "what is my address?", "Baker", "Elm"),
    ("I work as a teacher.", "I now work as a software engineer.", "what is my job?", "teacher", "software engineer"),
    ("I work at Microsoft.", "I now work at Google.", "where do I work?", "Microsoft", "Google"),
    ("My flight leaves at 9am on Friday.", "Actually my flight got moved to 11am.", "when is my flight?", "9am", "11am"),
    ("The wedding is on June 5th.", "The wedding has been postponed to August 12th.", "when is the wedding?", "June", "August"),
    ("I have 2 kids.", "I have 3 kids now, the baby arrived!", "how many kids do I have?", "2 kids", "3 kids"),
    ("My budget for the trip is 2000 dollars.", "My budget for the trip is now 3500 dollars.", "what's my trip budget?", "2000", "3500"),
    ("My phone number is 555-0100.", "My phone number changed to 555-0199.", "what is my phone number?", "0100", "0199"),
    ("The team meeting is at 3pm on Tuesdays.", "The team meeting is at 4pm now.", "when is the team meeting?", "3pm", "4pm"),
    ("My dentist appointment is on the 14th of June.", "My dentist appointment was rescheduled to the 20th of June.", "when is my dentist appointment?", "14th", "20th"),
    ("I'm 29 years old.", "I'm 30 years old as of today!", "how old am I?", "29", "30"),
    ("My manager is Sarah.", "My new manager is Tom.", "who is my manager?", "Sarah", "Tom"),
    ("My favorite band is Coldplay.", "My favorite band is Radiohead these days.", "what's my favorite band?", "Coldplay", "Radiohead"),
    ("I drive a Honda Civic.", "I sold the Honda, I now drive a Tesla.", "what car do I drive?", "Honda", "Tesla"),
    ("My rent is 1200 dollars a month.", "My rent went up, it is now 1450 dollars a month.", "how much is my rent?", "1200", "1450"),
    ("The project deadline is March 3rd.", "The project deadline got pushed back to March 17th.", "when is the project deadline?", "March 3rd", "March 17th"),
    ("I weigh 80 kg.", "I weigh 74 kg now after the diet.", "how much do I weigh?", "80", "74"),
    ("My cat is called Biscuit.", "Correction: my cat is called Muffin, Biscuit was my old cat.", "what is my cat's name?", "Biscuit", "Muffin"),
    ("Our office is in Berlin.", "Our office moved to Munich in May.", "where is our office?", "Berlin", "Munich"),
    ("My gym class is on Monday at 6pm.", "My gym class moved to Wednesday at 7pm.", "when is my gym class?", "Monday", "Wednesday"),
    ("My salary is 85k.", "I got a raise, my salary is now 95k.", "what is my salary?", "85k", "95k"),
    ("I'm vegetarian.", "I'm vegan now.", "what is my diet?", "vegetarian", "vegan"),
    ("My doctor is Dr Patel.", "I switched doctors, my doctor is now Dr Gomez.", "who is my doctor?", "Patel", "Gomez"),
    ("The party starts at 8pm.", "Change of plans: the party starts at 9pm instead.", "when does the party start?", "8pm", "9pm"),
    ("My son's school is Greenwood Elementary.", "My son's school is now Riverside Academy.", "what school does my son go to?", "Greenwood", "Riverside"),
    ("My password hint is Blue Moon.", "I updated my password hint to Red Sun.", "what is my password hint?", "Blue Moon", "Red Sun"),
    ("My landlord is Mr Brown.", "My landlord is now Mrs Green since the building was sold.", "who is my landlord?", "Brown", "Green"),
    ("I live in Austin with my partner.", "We relocated, I live in Denver now.", "where do I live?", "Austin", "Denver"),
    ("My checkup is on Thursday at 10am.", "The checkup is now on Friday at 2pm.", "when is my checkup?", "Thursday", "Friday"),
    ("My train leaves at 7:45.", "The train now leaves at 8:15 because of works.", "when does my train leave?", "7:45", "8:15"),
    ("I own 3 bikes.", "I own 2 bikes now, I sold one.", "how many bikes do I own?", "3 bikes", "2 bikes"),
    ("Mum's birthday dinner is at Luigi's.", "Mum's birthday dinner moved to Sakura instead.", "where is mum's birthday dinner?", "Luigi", "Sakura"),
    ("My best friend is Anna.", "My best friend is Maya these days.", "who is my best friend?", "Anna", "Maya"),
    ("The conference is in Lisbon this year.", "The conference changed venue, it is in Porto now.", "where is the conference?", "Lisbon", "Porto"),
    ("My running goal is 10 km.", "My running goal is now 21 km, a half marathon!", "what is my running goal?", "10 km", "21 km"),
    ("My Spanish class costs 40 euros.", "My Spanish class costs 55 euros now.", "how much is my Spanish class?", "40", "55"),
    ("I like sushi.", "I don't like sushi anymore.", "do I like sushi?", "I like sushi", "anymore"),
    ("My pickup time is 5:30pm.", "The pickup time is 6pm from now on.", "what is my pickup time?", "5:30", "6pm"),
    ("Our anniversary trip is to Rome.", "We changed the anniversary trip, it's to Vienna now.", "where is our anniversary trip?", "Rome", "Vienna"),
]

# Pairs that look alike but must not be linked as a conflict.
CONTROLS = [
    ("I have a dog named Rex.", "I went to the park with Rex yesterday."),
    ("I ran 5 km this morning.", "I ran 10 km today, new record!"),
    ("My sister lives in Lisbon.", "My brother lives in Madrid."),
    ("I visited Paris in 2019.", "I visited Rome in 2021."),
    ("I like sushi.", "I also like ramen."),
    ("I'm allergic to peanuts.", "I'm allergic to cats too."),
    ("The meeting is at 3pm.", "Lunch is at 1pm."),
    ("I read 20 pages last night.", "I read 35 pages tonight."),
    ("My flight to Tokyo leaves at 9am.", "My flight back from Tokyo leaves at 6pm."),
    ("My mom is a nurse.", "My dad is a pilot."),
    ("I bought apples for 3 dollars.", "I bought bread for 2 dollars."),
    ("We watched Inception on Friday.", "We watched Dune on Saturday."),
    ("I live in Pune.", "My parents live in Delhi."),
    ("I work at Microsoft.", "My wife works at Google."),
    ("My son is 7 years old.", "My daughter is 4 years old."),
    ("The bakery opens at 7am.", "The library opens at 9am."),
    ("I had 2 coffees this morning.", "I had 3 coffees yesterday."),
    ("I scored 82 on the first test.", "I scored 91 on the second test."),
    ("My favorite band is Coldplay.", "My favorite movie is Inception."),
    ("I now play guitar every evening.", "I play piano on weekends."),
    ("Tom's birthday is in May.", "Anna's birthday is in October."),
    ("I visited my aunt in Leeds.", "Now I'm heading to visit my uncle in York."),
    ("My old car was a Ford.", "I drive a Honda now."),  # true history, not a contradiction of a current fact
    ("Caroline: I went to a pride parade last week.", "Melanie: I went to a pottery class last week."),
    ("The new cafe on Main Street is great.", "The new bakery on Park Road is great."),
    ("I paid 50 dollars for the concert ticket.", "The museum ticket was 20 dollars."),
    ("I have 2 cats.", "My neighbour has 3 cats."),
    ("I got a new phone, it is a Pixel.", "My laptop is a ThinkPad."),
    ("Our first date was in Venice.", "Our honeymoon was in Bali."),
    ("I'm reading Dune right now.", "I just finished reading Emma."),
]

# Written after the detectors were tuned on the sets above and never tuned
# on: an honest estimate of how the rules generalise.
HELDOUT_UPDATES = [
    ("My yoga class is at 6:30pm.", "Heads up, yoga class is at 7pm now.", "when is yoga class?", "6:30", "7pm"),
    ("I'm based in Toronto.", "I'm based in Vancouver now.", "where am I based?", "Toronto", "Vancouver"),
    ("My internet plan is 100 Mbps.", "I upgraded, my internet plan is 500 Mbps now.", "what is my internet speed?", "100", "500"),
    ("The book club meets on Wednesday.", "The book club moved to Thursday evenings.", "when does book club meet?", "Wednesday", "Thursday"),
    ("My emergency contact is Priya.", "My emergency contact is now Rahul.", "who is my emergency contact?", "Priya", "Rahul"),
    ("Our wifi password is Sunflower22.", "We changed the wifi password to Maple77.", "what is the wifi password?", "Sunflower22", "Maple77"),
    ("My coach is Mike.", "My coach changed, it's Laura now.", "who is my coach?", "Mike", "Laura"),
    ("The bus to work takes 40 minutes.", "The bus to work takes 25 minutes now with the new route.", "how long is my bus ride?", "40", "25"),
    ("My daughter's piano lesson is at 4pm on Saturdays.", "Piano lesson got moved to 10am on Saturdays.", "when is the piano lesson?", "4pm", "10am"),
    ("I'm 172 cm tall.", "Actually I'm 175 cm tall, I measured again.", "how tall am I?", "172", "175"),
    ("My bank is HSBC.", "I switched banks, I'm with Monzo now.", "what bank do I use?", "HSBC", "Monzo"),
    ("The quarterly review is on April 10th.", "The quarterly review was postponed to April 24th.", "when is the quarterly review?", "April 10", "April 24"),
    ("My car insurance costs 90 dollars a month.", "My car insurance costs 120 dollars a month now.", "how much is my car insurance?", "90", "120"),
    ("I sleep about 6 hours a night.", "I sleep 8 hours a night now, much better.", "how much do I sleep?", "6 hours", "8 hours"),
    ("My favourite restaurant is Nando's.", "My favourite restaurant is Dishoom these days.", "what's my favourite restaurant?", "Nando", "Dishoom"),
    ("I work for Deloitte.", "I left Deloitte, I now work for KPMG.", "who do I work for?", "Deloitte", "KPMG"),
    ("The flat viewing is at 11am.", "The flat viewing was rescheduled to 3pm.", "when is the flat viewing?", "11am", "3pm"),
    ("My therapist is Dr Okafor.", "My new therapist is Dr Lindqvist.", "who is my therapist?", "Okafor", "Lindqvist"),
    ("We have 1 dog.", "We have 2 dogs now, we adopted another one!", "how many dogs do we have?", "1 dog", "2 dogs"),
    ("My commute is by car.", "I commute by train now instead.", "how do I commute?", "car", "train"),
]

HELDOUT_CONTROLS = [
    ("My yoga class is at 6:30pm.", "My pottery class is at 8pm."),
    ("I'm based in Toronto.", "My cousin is based in Montreal."),
    ("The book club meets on Wednesday.", "The chess club meets on Friday."),
    ("I visited Kyoto in 2018.", "I now want to visit Osaka next year."),
    ("My emergency contact is Priya.", "My manager is Rahul."),
    ("I spent 40 dollars on groceries.", "I spent 60 dollars on groceries this week."),
    ("I'm 172 cm tall.", "My brother is 180 cm tall."),
    ("I had sushi with Sam on Monday.", "I had pizza with Jo on Tuesday."),
    ("We have 1 dog.", "Our neighbours have 2 dogs."),
    ("The flat viewing is at 11am.", "The car viewing is at 3pm."),
    ("I watched 3 episodes last night.", "I watched 5 episodes tonight."),
    ("Our wifi password is Sunflower22.", "Our office door code is 4471."),
    ("I work for Deloitte.", "My friend Ana works for KPMG now."),
    ("I sleep about 6 hours a night.", "The baby sleeps 12 hours a night."),
    ("I booked a table at Dishoom for Friday.", "I booked a table at Nando's for Sunday."),
    ("I'm learning French.", "I'm learning to cook Thai food now."),
    ("My bus to work takes 40 minutes.", "The train to Leeds takes 2 hours."),
    ("I went to Berlin in March.", "I'm going to Madrid in June."),
    ("Sara is turning 30 in July.", "Ben is turning 32 in September."),
    ("My first marathon took 4 hours.", "My second marathon took 3 hours 50 minutes."),
]

# Fresh held-out set, written before the second round of rule changes
# (plurals, weekday plans, value-less changes) and never tuned on. The set
# above became a development set once its misses were fixed.
HELDOUT2_UPDATES = [
    ("I take the bus to university.", "I cycle to university now instead.", "how do I get to university?", "bus", "cycle"),
    ("My standup is on Mondays at 9am.", "Standup moved to Tuesdays.", "when is my standup?", "Monday", "Tuesday"),
    ("I own one car.", "I own two cars now.", "how many cars do I own?", "one car", "two cars"),
    ("My go-to drink is tea.", "My go-to drink is coffee these days.", "what do I drink?", "tea", "coffee"),
    ("The farmers market is on Saturday.", "The farmers market switched to Sunday.", "when is the farmers market?", "Saturday", "Sunday"),
    ("I use an iPhone.", "I switched to an Android phone, I use a Pixel now.", "what phone do I use?", "iPhone", "Pixel"),
    ("Our team has 6 engineers.", "Our team has 9 engineers now after hiring.", "how big is our team?", "6", "9"),
    ("I get paid monthly.", "I get paid weekly now.", "how often am I paid?", "monthly", "weekly"),
    ("My parents live in Chennai.", "My parents moved to Mysore.", "where do my parents live?", "Chennai", "Mysore"),
    ("Swimming lessons are on Thursday afternoons.", "Swimming lessons are on Friday afternoons from now on.", "when are swimming lessons?", "Thursday", "Friday"),
    ("I work from the office.", "I work from home now.", "where do I work from?", "office", "home"),
    ("The rent is due on the 1st.", "The rent is now due on the 5th.", "when is rent due?", "1st", "5th"),
    ("My partner's name is Alex.", "Actually my partner's name is Alexis, Alex is a nickname.", "what is my partner's name?", "Alex", "Alexis"),
    ("I have one sibling.", "I have two siblings, I found out about a half-brother.", "how many siblings do I have?", "one sibling", "two siblings"),
    ("My preferred editor is Vim.", "My preferred editor is VS Code now.", "what editor do I use?", "Vim", "VS Code"),
    ("I shop at Tesco.", "I shop at Aldi now, it's cheaper.", "where do I shop?", "Tesco", "Aldi"),
    ("The dinner reservation is for 4 people.", "The dinner reservation is for 6 people now.", "how many people is the reservation for?", "4", "6"),
    ("My laptop runs Windows.", "My laptop runs Linux now.", "what does my laptop run?", "Windows", "Linux"),
    ("I water the plants every Sunday.", "I water the plants every Wednesday now.", "when do I water the plants?", "Sunday", "Wednesday"),
    ("I usually wake up at 7am.", "I wake up at 5:30am now for training.", "when do I wake up?", "7am", "5:30"),
    ("My hair is long.", "My hair is short now, I cut it.", "how is my hair?", "long", "short"),
    ("Grandma's surgery is on 3 December.", "Grandma's surgery was moved to 10 December.", "when is grandma's surgery?", "3 December", "10 December"),
    ("We have two cats.", "We have three cats now.", "how many cats do we have?", "two cats", "three cats"),
    ("My office is on the 3rd floor.", "My office moved to the 7th floor.", "which floor is my office?", "3rd", "7th"),
    ("The newsletter goes out on Tuesday.", "The newsletter goes out on Thursday now.", "when does the newsletter go out?", "Tuesday", "Thursday"),
]

HELDOUT2_CONTROLS = [
    ("I take the bus to university.", "My roommate cycles to university."),
    ("My standup is on Mondays at 9am.", "My retro is on Fridays at 4pm."),
    ("I own one car.", "My parents own two cars."),
    ("I drank tea this morning.", "I drank coffee this afternoon."),
    ("The farmers market is on Saturday.", "The flea market is on Sunday."),
    ("I work from the office on Mondays.", "I work from home on Fridays."),
    ("My hair is long.", "My sister's hair is short now."),
    ("I water the plants every Sunday.", "I clean the bathroom every Wednesday."),
    ("I woke up at 7am today.", "I woke up at 9am yesterday."),
    ("We have two cats.", "Our friends have three dogs now."),
    ("I shop at Tesco for groceries.", "I shop at Ikea for furniture."),
    ("My laptop runs Windows.", "My work desktop runs Linux."),
    ("I visited Chennai in 2020.", "Now I'm visiting Mysore for the weekend."),
    ("The rent is due on the 1st.", "The electricity bill is due on the 15th."),
    ("I used Vim for this project.", "For the website I used VS Code."),
    ("I ate pasta on Monday.", "I ate curry on Tuesday."),
    ("My office is on the 3rd floor.", "The cafeteria is on the 1st floor."),
    ("I have one sibling.", "My husband has four siblings."),
    ("I read the news in the morning.", "I read novels in the evening."),
    ("The team lunch was at Wagamama.", "The client dinner was at Hawksmoor."),
    ("I'm learning Python.", "I'm also learning Go now."),
    ("My bus pass costs 60 pounds.", "My gym membership costs 40 pounds."),
    ("The dinner reservation is for 4 people.", "The lunch booking is for 2 people."),
    ("I prefer window seats on trains.", "I prefer aisle seats on planes."),
    ("I cycled 20 km on Saturday.", "I cycled 35 km on Sunday."),
]

FILLER = [
    "That sounds fun, tell me more.", "Haha yes, totally agree.", "The weather was lovely today.",
    "I watched a documentary about whales.", "Did you try the new recipe?", "Work has been busy lately.",
    "We should catch up soon.", "I need to buy groceries later.", "The traffic this morning was awful.",
    "Have you seen the latest episode?", "My plants are growing nicely.", "I'm thinking about learning to paint.",
    "Coffee first, then emails.", "The game last night was close.", "I cleaned the whole kitchen.",
    "Let me know what you think.", "That book was better than the film.", "I'm so tired this week.",
]

DAY = 86400
T0 = datetime(2026, 1, 5, 9, 0).timestamp()


def _session(old: str, new: str, rng: random.Random, mem_kwargs) -> tuple[NoMem, object, object]:
    mem = NoMem(max_tokens=400, counter=ApproxCounter(), core="You are helpful.", **mem_kwargs)
    t = T0
    for _ in range(rng.randint(3, 8)):
        mem.add_user(rng.choice(FILLER), timestamp=t)
        t += 600
    a = mem.add_user(old, timestamp=t)
    t += DAY * rng.randint(5, 40)
    for _ in range(40):
        mem.add("user" if rng.random() < 0.6 else "assistant", rng.choice(FILLER), timestamp=t)
        t += 600
    b = mem.add_user(new, timestamp=t)
    t += DAY
    for _ in range(30):
        mem.add("user" if rng.random() < 0.6 else "assistant", rng.choice(FILLER), timestamp=t)
        t += 600
    return mem, a, b


def _linked(mem: NoMem, a, b) -> list[dict]:
    return [c for c in mem.conflicts() if {c["older_id"], c["newer_id"]} == {a.id, b.id}]


def run(mem_kwargs=None, verbose: bool = False, locomo: bool = True) -> dict:
    mem_kwargs = mem_kwargs or {}
    out = {"tuned": evaluate(UPDATES, CONTROLS, mem_kwargs, verbose),
           "dev": evaluate(HELDOUT_UPDATES, HELDOUT_CONTROLS, mem_kwargs, verbose),
           "heldout": evaluate(HELDOUT2_UPDATES, HELDOUT2_CONTROLS, mem_kwargs, verbose)}
    if locomo:
        out["locomo"] = locomo_flags(mem_kwargs, verbose)
    return out


def evaluate(UPDATES, CONTROLS, mem_kwargs, verbose: bool = False) -> dict:
    rng = random.Random(7)
    found = surfaced = 0
    kinds: dict = {}
    misses = []
    for old, new, q, n_old, n_new in UPDATES:
        mem, a, b = _session(old, new, rng, mem_kwargs)
        link = [c for c in _linked(mem, a, b) if c["older_id"] == a.id]
        if link:
            found += 1
            kinds[link[0]["kind"]] = kinds.get(link[0]["kind"], 0) + 1
        else:
            misses.append(new)
        text = mem.build_context(query=q).text
        ok = n_old in text and n_new in text and "CONFLICT" in text
        surfaced += ok
        if verbose and (not link or not ok):
            print(f"[update {'miss' if not link else 'not surfaced'}] {old!r} -> {new!r}\n{text}\n")
        mem.close()

    false_flags = []
    for x, y in CONTROLS:
        mem, a, b = _session(x, y, rng, mem_kwargs)
        if _linked(mem, a, b):
            false_flags.append((x, y, _linked(mem, a, b)[0]["detail"]))
        mem.close()
    if verbose:
        for f in false_flags:
            print("[false flag]", f)

    out = {
        "updates": len(UPDATES),
        "detected": found,
        "detection_recall": round(100 * found / len(UPDATES), 1),
        "by_kind": kinds,
        "both_surfaced": round(100 * surfaced / len(UPDATES), 1),
        "controls": len(CONTROLS),
        "false_flags": len(false_flags),
        "false_flag_rate": round(100 * len(false_flags) / len(CONTROLS), 1),
        "missed": misses,
        "false_flag_pairs": [f"{x} | {y}" for x, y, _ in false_flags],
    }
    return out


def locomo_flags(mem_kwargs, verbose: bool = False, sample: int = 15) -> dict:
    import locomo_bench

    if not locomo_bench.DEFAULT_DATA.exists():
        return {}
    data = json.loads(locomo_bench.DEFAULT_DATA.read_text())
    n_msgs, rows = 0, []
    for i, conv in enumerate(data):
        mem = NoMem(max_tokens=1000, counter=ApproxCounter(), session_id=f"conv{i}", **mem_kwargs)
        by_dia, _ = locomo_bench.ingest(mem, conv["conversation"])
        n_msgs += len(by_dia)
        msgs = mem.store.get_messages([c["older_id"] for c in mem.conflicts()] + [c["newer_id"] for c in mem.conflicts()])
        for c in mem.conflicts():
            rows.append((c["kind"], c["confidence"], msgs[c["older_id"]].content, msgs[c["newer_id"]].content, c["detail"]))
        mem.close()
    by_kind: dict = {}
    for r in rows:
        by_kind[r[0]] = by_kind.get(r[0], 0) + 1
    rng = random.Random(3)
    picked = rng.sample(rows, min(sample, len(rows)))
    if verbose:
        for k, conf, o, n, d in picked:
            print(f"[locomo {k} {conf}] {d}\n   old: {o[:140]}\n   new: {n[:140]}")
    return {
        "messages": n_msgs,
        "conflicts": len(rows),
        "per_1k_messages": round(1000 * len(rows) / max(1, n_msgs), 1),
        "by_kind": by_kind,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--embedder")
    ap.add_argument("--no-locomo", action="store_true")
    args = ap.parse_args()
    kw = {}
    if args.embedder:
        sys.path.insert(0, str(Path(__file__).parent))
        from run_all import make_embedder
        kw["embedder"] = make_embedder(args.embedder)
    print(json.dumps(run(kw, args.verbose, not args.no_locomo), indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
