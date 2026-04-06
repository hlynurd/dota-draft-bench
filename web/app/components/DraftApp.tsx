"use client";

import { useState, useMemo } from "react";
import type { OpenDotaHero } from "./HeroPicker";
import HeroPicker from "./HeroPicker";
import { type DraftData, type IndexedDraftData, type ItemRec, indexDraftData, recommendItems } from "@/lib/compute";
import { heroImgUrl, itemImgUrl } from "@/lib/cdn";

interface Props {
  heroes: OpenDotaHero[];
  draftData: DraftData | null;
  itemNames: Record<number, { name: string; dname: string }>;
}

type Side = "radiant" | "dire";

export default function DraftApp({ heroes, draftData, itemNames }: Props) {
  const [radiant, setRadiant] = useState<(OpenDotaHero | null)[]>(Array(5).fill(null));
  const [dire, setDire] = useState<(OpenDotaHero | null)[]>(Array(5).fill(null));
  const [picker, setPicker] = useState<{ side: Side; slot: number } | null>(null);

  const indexed = useMemo(() => draftData ? indexDraftData(draftData) : null, [draftData]);
  const itemNameMap = useMemo(() => new Map(Object.entries(itemNames).map(([k, v]) => [Number(k), v.dname])), [itemNames]);
  const itemInternalMap = useMemo(() => new Map(Object.entries(itemNames).map(([k, v]) => [Number(k), v.name])), [itemNames]);

  const selectedIds = new Set([...radiant, ...dire].filter(Boolean).map(h => h!.id));

  // Compute recs for all heroes with a filled draft
  const recs = useMemo(() => {
    if (!indexed) return new Map<number, ItemRec[]>();
    const result = new Map<number, ItemRec[]>();

    const sides: [typeof radiant, typeof dire][] = [[radiant, dire], [dire, radiant]];
    for (const [sideHeroes, opponents] of sides) {
      const enemyIds = opponents.filter(Boolean).map(h => h!.id);
      const filledHeroes = sideHeroes.filter(Boolean) as OpenDotaHero[];

      for (const hero of filledHeroes) {
        const allyIds = filledHeroes.filter(h => h.id !== hero.id).map(h => h.id);
        if (enemyIds.length === 0 && allyIds.length === 0) continue;
        const items = recommendItems(indexed, hero.id, allyIds, enemyIds, itemNameMap);
        result.set(hero.id, items.slice(0, 12));
      }
    }
    return result;
  }, [indexed, radiant, dire, itemNameMap]);

  function handleSelect(hero: OpenDotaHero) {
    if (!picker) return;
    const { side, slot } = picker;
    if (side === "radiant") setRadiant(prev => prev.map((h, i) => i === slot ? hero : h));
    else setDire(prev => prev.map((h, i) => i === slot ? hero : h));
    setPicker(null);
  }

  function clearSlot(side: Side, slot: number) {
    if (side === "radiant") setRadiant(prev => prev.map((h, i) => i === slot ? null : h));
    else setDire(prev => prev.map((h, i) => i === slot ? null : h));
  }

  function HeroSlot({ hero, side, slot }: { hero: OpenDotaHero | null; side: Side; slot: number }) {
    const accent = side === "radiant" ? "text-green-400 border-green-800 hover:border-green-600" : "text-red-400 border-red-900 hover:border-red-700";
    if (!hero) {
      return (
        <button onClick={() => setPicker({ side, slot })}
          className={`h-10 rounded-lg border border-dashed border-zinc-700 hover:border-zinc-500 flex items-center justify-center text-xs ${accent.split(" ")[0]}`}>
          + Hero
        </button>
      );
    }
    return (
      <div className={`h-10 rounded-lg border ${accent} bg-zinc-900 flex items-center gap-2 px-2 transition-colors`}>
        <button onClick={() => setPicker({ side, slot })} className="shrink-0 w-14 h-7 rounded overflow-hidden">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={heroImgUrl(hero.name)} alt={hero.localized_name} className="w-full h-full object-cover object-top" loading="lazy" />
        </button>
        <span className="text-xs truncate flex-1">{hero.localized_name}</span>
        <button onClick={() => clearSlot(side, slot)} className="text-zinc-600 hover:text-zinc-300 text-sm">×</button>
      </div>
    );
  }

  function ItemCard({ rec }: { rec: ItemRec }) {
    const name = itemNameMap.get(rec.item_id) ?? "?";
    const internal = itemInternalMap.get(rec.item_id) ?? "";
    const diffColor = rec.wr_diff >= 0.005 ? "text-green-400" : rec.wr_diff <= -0.005 ? "text-red-400" : "text-zinc-500";
    const diffSign = rec.wr_diff >= 0 ? "+" : "";
    return (
      <div className="flex items-center gap-2 py-1 border-b border-zinc-800/50 last:border-0">
        <div className="w-7 h-5 rounded overflow-hidden bg-zinc-800 shrink-0">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={itemImgUrl(internal)} alt={name} className="w-full h-full object-cover" loading="lazy"
            onError={e => { (e.target as HTMLImageElement).style.display = "none"; }} />
        </div>
        <span className="text-xs text-zinc-300 truncate flex-1">{name}</span>
        <span className={`text-xs font-mono shrink-0 w-12 text-right ${diffColor}`}>
          {diffSign}{(rec.wr_diff * 100).toFixed(1)}%
        </span>
      </div>
    );
  }

  function HeroRecs({ hero }: { hero: OpenDotaHero }) {
    const items = recs.get(hero.id);
    if (!items || items.length === 0) return null;
    return (
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-8 h-5 rounded overflow-hidden">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={heroImgUrl(hero.name)} alt={hero.localized_name} className="w-full h-full object-cover object-top" loading="lazy" />
          </div>
          <span className="text-sm font-medium">{hero.localized_name}</span>
        </div>
        {items.map(rec => <ItemCard key={rec.item_id} rec={rec} />)}
      </div>
    );
  }

  const allHeroes = [...radiant, ...dire].filter(Boolean) as OpenDotaHero[];

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-zinc-800 px-6 py-3">
        <div className="max-w-5xl mx-auto flex items-center gap-3">
          <div className="w-2 h-6 bg-red-500 rounded-sm" />
          <h1 className="text-xl font-semibold tracking-tight">Dota 2 Draft Items</h1>
          <span className="ml-2 text-xs text-zinc-500 font-mono">beta</span>
        </div>
      </header>

      <main className="flex-1 max-w-5xl mx-auto w-full p-4 flex flex-col gap-6">
        {/* Draft input */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <span className="text-xs font-semibold uppercase tracking-widest text-green-400">Radiant</span>
            </div>
            <div className="flex flex-col gap-1.5">
              {radiant.map((hero, i) => <HeroSlot key={i} hero={hero} side="radiant" slot={i} />)}
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 rounded-full bg-red-500" />
              <span className="text-xs font-semibold uppercase tracking-widest text-red-400">Dire</span>
            </div>
            <div className="flex flex-col gap-1.5">
              {dire.map((hero, i) => <HeroSlot key={i} hero={hero} side="dire" slot={i} />)}
            </div>
          </div>
        </div>

        {/* Item recommendations per hero */}
        {allHeroes.length > 0 && recs.size > 0 && (
          <>
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-widest text-zinc-500">Recommended Items</span>
              <div className="flex-1 h-px bg-zinc-800" />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {allHeroes.map(hero => <HeroRecs key={hero.id} hero={hero} />)}
            </div>
          </>
        )}
      </main>

      {picker && (
        <HeroPicker heroes={heroes} excludeIds={selectedIds}
          onSelect={handleSelect} onClose={() => setPicker(null)} />
      )}
    </div>
  );
}
