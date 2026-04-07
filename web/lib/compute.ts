/**
 * Client-side draft-context item recommendation.
 * Loads draft-data.json, indexes it, computes per-hero recommendations.
 */

export interface DraftData {
  /** [buyer_hero, item, context_hero, side(0=enemy,1=ally), games, wins] */
  p: [number, number, number, number, number, number][];
  /** [buyer_hero, item, bought, won] */
  b: [number, number, number, number][];
  /** [hero_id, games, wins] */
  h: [number, number, number][];
  ts: number;
}

export interface ItemRec {
  item_id: number;
  buy_rate_lift: number;  // how much more likely vs baseline
  wr_diff: number;        // P(win|buy) - P(win|not buy) in context
  excess_wr: number;      // wr_diff minus item's global avg WR diff (removes gold bias)
  zscore: number;         // std deviations from item's mean WR diff across matchups
  wr_with: number;
  wr_without: number;
  games: number;          // sample size in context
}

interface PairwiseEntry {
  games: number;
  wins: number;
}

export interface IndexedDraftData {
  // (buyer, item, context, side) -> {games, wins}
  pairwise: Map<string, PairwiseEntry>;
  // (buyer, item) -> {bought, won}
  baselines: Map<string, { bought: number; won: number }>;
  // hero -> {games, wins}
  heroTotals: Map<number, { games: number; wins: number }>;
  // (buyer, item) -> avg pairwise rate across all context heroes for this side
  baselineRates: Map<string, { enemy: number; ally: number }>;
  // (buyer, item) -> global WR diff (no context): P(win|buy) - P(win|not buy)
  globalWrDiff: Map<string, number>;
  // (buyer, item) -> stddev of pairwise WR diffs across all context heroes
  wrDiffStd: Map<string, number>;
}

export function indexDraftData(raw: DraftData): IndexedDraftData {
  const pairwise = new Map<string, PairwiseEntry>();
  const baselines = new Map<string, { bought: number; won: number }>();
  const heroTotals = new Map<number, { games: number; wins: number }>();

  for (const [bh, item, ch, side, games, wins] of raw.p) {
    pairwise.set(`${bh}:${item}:${ch}:${side}`, { games, wins });
  }
  for (const [bh, item, bought, won] of raw.b) {
    baselines.set(`${bh}:${item}`, { bought, won });
  }
  for (const [hid, games, wins] of raw.h) {
    heroTotals.set(hid, { games, wins });
  }

  // Pre-compute baseline purchase rates per (buyer, item) per side
  const baselineRates = new Map<string, { enemy: number; ally: number }>();
  // (buyer, item) -> sum of rates per side, count per side
  const rateAccum = new Map<string, { eSum: number; eCount: number; aSum: number; aCount: number }>();
  for (const [bh, item, ch, side, games] of raw.p) {
    const key = `${bh}:${item}`;
    const ht = heroTotals.get(ch);
    if (!ht || ht.games === 0) continue;
    const rate = games / ht.games;
    let acc = rateAccum.get(key);
    if (!acc) { acc = { eSum: 0, eCount: 0, aSum: 0, aCount: 0 }; rateAccum.set(key, acc); }
    if (side === 0) { acc.eSum += rate; acc.eCount++; }
    else { acc.aSum += rate; acc.aCount++; }
  }
  for (const [key, acc] of rateAccum) {
    baselineRates.set(key, {
      enemy: acc.eCount > 0 ? acc.eSum / acc.eCount : 0,
      ally: acc.aCount > 0 ? acc.aSum / acc.aCount : 0,
    });
  }

  // Pre-compute global WR diff per (buyer, item): P(win|buy) - P(win|not buy)
  const globalWrDiff = new Map<string, number>();
  for (const [key, { bought, won }] of baselines) {
    const [bh] = key.split(":");
    const ht = heroTotals.get(Number(bh));
    if (!ht || ht.games === 0 || bought === 0) continue;
    const wrWith = won / bought;
    const withoutGames = ht.games - bought;
    const withoutWins = ht.wins - won;
    const wrWithout = withoutGames > 0 ? withoutWins / withoutGames : 0.5;
    globalWrDiff.set(key, wrWith - wrWithout);
  }

  // Pre-compute stddev of pairwise WR diffs per (buyer, item)
  const wrDiffStd = new Map<string, number>();
  const wrDiffAccum = new Map<string, number[]>();
  for (const [bh, item, ch, , games, wins] of raw.p) {
    if (games < 3) continue;
    const ht = heroTotals.get(Number(bh));
    if (!ht || ht.games === 0) continue;
    const bl = baselines.get(`${bh}:${item}`);
    if (!bl || bl.bought === 0) continue;
    const wrWith = wins / games;
    const withoutGames = ht.games - bl.bought;
    const withoutWins = ht.wins - bl.won;
    const wrWithout = withoutGames > 0 ? withoutWins / withoutGames : 0.5;
    const diff = wrWith - wrWithout;
    const key = `${bh}:${item}`;
    let arr = wrDiffAccum.get(key);
    if (!arr) { arr = []; wrDiffAccum.set(key, arr); }
    arr.push(diff);
  }
  for (const [key, diffs] of wrDiffAccum) {
    if (diffs.length < 2) continue;
    const mean = diffs.reduce((s, d) => s + d, 0) / diffs.length;
    const variance = diffs.reduce((s, d) => s + (d - mean) ** 2, 0) / diffs.length;
    wrDiffStd.set(key, Math.sqrt(variance));
  }

  return { pairwise, baselines, heroTotals, baselineRates, globalWrDiff, wrDiffStd };
}

/**
 * Recommend items for a hero given the full draft.
 */
export function recommendItems(
  data: IndexedDraftData,
  buyerHero: number,
  allies: number[],    // 4 other heroes on buyer's team
  enemies: number[],   // 5 heroes on enemy team
  itemNames: Map<number, string>,  // item_id -> display name
): ItemRec[] {
  const heroTotal = data.heroTotals.get(buyerHero);
  if (!heroTotal || heroTotal.games === 0) return [];
  const totalGames = heroTotal.games;
  const totalWins = heroTotal.wins;

  const results: ItemRec[] = [];

  // For each item this hero buys
  for (const [key, { bought, won }] of data.baselines) {
    const [bh, itemStr] = key.split(":");
    if (Number(bh) !== buyerHero) continue;
    const item_id = Number(itemStr);

    const baseRate = bought / totalGames;
    if (baseRate < 0.01) continue; // skip very rare items

    // Compute context-adjusted buy rate and win rate
    let contextGames = 0;
    let contextWins = 0;
    let buyRateProduct = 1.0;
    let hasContext = false;

    for (const enemy of enemies) {
      const pw = data.pairwise.get(`${buyerHero}:${item_id}:${enemy}:0`);
      if (pw && pw.games >= 3) {
        const ht = data.heroTotals.get(enemy);
        if (ht && ht.games > 0) {
          const pairRate = pw.games / ht.games;
          const baseline = data.baselineRates.get(`${buyerHero}:${item_id}`);
          const avgRate = baseline?.enemy || baseRate;
          if (avgRate > 0) buyRateProduct *= (pairRate / avgRate);
          hasContext = true;
        }
        contextGames += pw.games;
        contextWins += pw.wins;
      }
    }

    for (const ally of allies) {
      const pw = data.pairwise.get(`${buyerHero}:${item_id}:${ally}:1`);
      if (pw && pw.games >= 3) {
        const ht = data.heroTotals.get(ally);
        if (ht && ht.games > 0) {
          const pairRate = pw.games / ht.games;
          const baseline = data.baselineRates.get(`${buyerHero}:${item_id}`);
          const avgRate = baseline?.ally || baseRate;
          if (avgRate > 0) buyRateProduct *= (pairRate / avgRate);
          hasContext = true;
        }
        contextGames += pw.games;
        contextWins += pw.wins;
      }
    }

    // WR with and without
    const wrWith = contextGames > 0 ? contextWins / contextGames : won / bought;
    const withoutGames = totalGames - bought;
    const withoutWins = totalWins - won;
    const wrWithout = withoutGames > 0 ? withoutWins / withoutGames : 0.5;

    const wrDiff = wrWith - wrWithout;
    const baselineKey = `${buyerHero}:${item_id}`;
    const gDiff = data.globalWrDiff.get(baselineKey) ?? 0;
    const std = data.wrDiffStd.get(baselineKey) ?? 0;
    const excessWr = wrDiff - gDiff;
    const zscore = std > 0.001 ? excessWr / std : 0;

    results.push({
      item_id,
      buy_rate_lift: hasContext ? Math.round(buyRateProduct * 100) / 100 : 1.0,
      wr_diff: Math.round(wrDiff * 10000) / 10000,
      excess_wr: Math.round(excessWr * 10000) / 10000,
      zscore: Math.round(zscore * 100) / 100,
      wr_with: Math.round(wrWith * 10000) / 10000,
      wr_without: Math.round(wrWithout * 10000) / 10000,
      games: contextGames || bought,
    });
  }

  // Sort by excess_wr (removes gold/cost bias), weighted by confidence
  results.sort((a, b) => {
    const scoreA = a.excess_wr * Math.min(1, a.games / 50);
    const scoreB = b.excess_wr * Math.min(1, b.games / 50);
    return scoreB - scoreA;
  });

  return results;
}
