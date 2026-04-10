# Case Study: How Teammates Shape Item Choices

*Based on 1.39M ranked matches from Dota 2 patch 7.41a (Valve Steam API, April 2026)*

Your allies change what you buy. Not just through vague "synergy" — the data shows stark, double-digit swings in purchase rates driven by your team composition. Here are four patterns where teammates move the needle on item choices, with data and reasoning.

---

## 1. The Alchemist Effect: "Don't Buy Aghs, I'll Give You One"

Alchemist can spend 4200 gold to gift an Aghanim's Scepter to a teammate as a permanent buff. The gifted Aghs doesn't occupy an item slot — it's consumed. This means the recipient's end-game inventory (what the Valve API records) never shows the Aghs.

The result is one of the largest ally effects in the dataset: **teammates universally stop buying Aghs when Alchemist is on the team.**

| Hero | Aghs rate (no Alch) | Aghs rate (w/ Alch) | Change |
|------|:---:|:---:|:---:|
| Pangolier | 72.0% | 16.5% | **-55.5pp** |
| Void Spirit | 65.8% | 10.9% | **-54.9pp** |
| Primal Beast | 62.5% | 11.8% | **-50.7pp** |
| Sand King | 73.5% | 23.6% | **-49.8pp** |
| Queen of Pain | 52.2% | 8.7% | **-43.5pp** |

**Why the residual 10-20%?** Games where Alchemist doesn't have enough farm to gift Aghs (lost lanes, short games), or where the hero buys their own Aghs before Alchemist can gift it. Heroes like Beastmaster still self-purchase at 39.5% because their Aghs is so critical they can't wait.

**Insight for the model:** This is a *substitution effect* — the item isn't gone from the game, it's just acquired through a different channel. A recommendation engine should know that Aghs is lower priority for heroes with an Alchemist ally, not because it's bad, but because it's likely coming for free.

---

## 2. Lina's Identity Crisis: Support or Core?

Lina is one of Dota's most flexible heroes — she can play mid (core) or position 4/5 (support). Her item build reveals which role she's in:

- **Core Lina**: Blink Dagger (63.6% base rate), damage items
- **Support Lina**: Arcane Boots (15.2% base rate), utility items

Her teammates determine which Lina shows up:

| Ally | Arcane Boots | Blink Dagger | Role signal |
|------|:---:|:---:|---|
| Arc Warden | 42.2% | 32.5% | Support Lina |
| Ember Spirit | 42.0% | 30.9% | Support Lina |
| Storm Spirit | 40.1% | 31.3% | Support Lina |
| Puck | 37.1% | 31.9% | Support Lina |
| Void Spirit | 36.8% | 37.9% | Support Lina |
| ... | | | |
| Crystal Maiden | 9.1% | 70.0% | Core Lina |
| Venomancer | 9.8% | 70.4% | Core Lina |
| Shadow Shaman | 10.0% | 69.5% | Core Lina |

**The pattern:** When Lina's team already has a flashy mid hero (Ember, Storm, Puck, Arc Warden), she's drafted as support — and her Arcane Boots rate triples from 15% to 40%. When her team has traditional supports (CM, Veno, Shadow Shaman), she's the core — and Blink Dagger jumps from 64% to 70%.

**Insight:** This isn't an item preference — it's a *role signal*. The ally composition doesn't make Lina want different items; it determines whether she's playing a fundamentally different position. A draft-aware recommender needs to infer the hero's role from context, not just their identity.

---

## 3. Bloodseeker's Blade Mail: "I Need Someone to Hit Me"

Bloodseeker's passive heals him when nearby enemies die, and his Rupture punishes enemies for moving. Blade Mail (reflects damage back to attackers) is a natural fit — but only if enemies actually want to attack him. His Blade Mail rate swings from 10% to 41% depending on allies:

**Allies that increase Blade Mail (20.9% → 35-41%):**
- Juggernaut (41.3%), Phantom Assassin (39.5%), Drow Ranger (39.1%)
- Faceless Void (39.0%), Anti-Mage (37.2%), Spectre (37.1%)

**Allies that decrease Blade Mail (20.9% → 10-11%):**
- Broodmother (10.0%), Pangolier (10.2%), Tinker (10.7%)

**The reasoning:** When Bloodseeker's team has hard carry allies (Jugg, PA, Drow, Void, AM), he's playing position 3 offlane — an initiator/frontliner role where Blade Mail is core. He runs in, forces enemies to hit him (Blade Mail reflects), and his carries clean up.

When his team has Broodmother, Pangolier, or Tinker — heroes that dominate lanes and take farm priority — Bloodseeker is played as a greedy support or roamer. In that role, he can't afford Blade Mail and doesn't want to be in the front line.

**Insight:** Like Lina, this is a role-dependent pattern. But the item itself (Blade Mail) makes strategic sense specifically because of the team composition — the carries create a "hit me instead" dynamic that Blade Mail exploits.

---

## 4. The Sustain Gap: Who Builds Pipe and Greaves?

Pipe of Insight (team magic barrier) and Guardian Greaves (team heal + dispel) are "someone has to buy this" items. They're expensive, slot-inefficient, and rarely desired — but essential against certain enemy compositions. The question is *who* ends up buying them.

**Pipe of Insight** (3.7% base rate, some heroes 50%+):

| Hero | Ally | Pipe rate | Base | Change |
|------|------|:---------:|:----:|:------:|
| Underlord | Hoodwink | 59.0% | 3.7% | +55.3pp |
| Underlord | Pangolier | 55.1% | 3.7% | +51.4pp |
| Underlord | Winter Wyvern | 55.0% | 3.7% | +51.3pp |
| Underlord | Tinker | 53.9% | 3.7% | +50.2pp |

**Guardian Greaves** (5.0% base rate):

| Hero | Ally | Greaves rate | Base | Change |
|------|------|:------------:|:----:|:------:|
| Dark Seer | Puck | 63.0% | 5.0% | +58.0pp |
| Chen | Weaver | 62.3% | 5.0% | +57.3pp |
| Dark Seer | Tusk | 61.9% | 5.0% | +56.9pp |
| Chen | Tiny | 60.8% | 5.0% | +55.8pp |

**The pattern:** Underlord, Dark Seer, and Chen are natural aura carriers — tanky heroes who stand in the middle of fights. When they're on teams with heroes that *can't* build these items (mobile assassins like Hoodwink, Pangolier, Puck — or fragile supports like Winter Wyvern), the duty falls squarely on them.

The 0% rates at the bottom tell the inverse story: when Drow Ranger or Juggernaut are on your team, nobody builds Pipe or Greaves — these are fast-push or right-click lineups that win through damage, not sustain.

**Insight:** This is a *team composition gap-filling* pattern. The model shouldn't just ask "is this item good for this hero?" but "does this team need this item, and is this hero the one who should build it?"

---

## Summary: Three Types of Ally Effects

1. **Substitution** (Alchemist + Aghs): The item is still acquired, just through a different mechanism. The model needs to understand alternative acquisition channels.

2. **Role determination** (Lina's build, Bloodseeker's Blade Mail): Allies determine *what position* the hero plays, which cascades into completely different item builds. The model needs to infer role from draft context.

3. **Gap-filling** (Pipe, Greaves): Certain items must be built by *someone* on the team. Allies determine who gets stuck with the bill. The model needs team-level reasoning about unmet needs.

All three patterns are invisible to a hero-only model. They require understanding the full draft to make useful recommendations.

---

*Data: 1.39M ranked All Pick matches, patch 7.41a, minimum 200 games per (hero, item, ally) triple.*
*Analysis: [dota-draft-bench](https://github.com/hlynurd/dota-draft-bench)*
