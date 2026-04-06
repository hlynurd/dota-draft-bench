import { readFileSync, existsSync } from "fs";
import { join } from "path";
import type { DraftData } from "@/lib/compute";
import type { OpenDotaHero } from "./components/HeroPicker";
import DraftApp from "./components/DraftApp";

async function fetchHeroes(): Promise<OpenDotaHero[]> {
  try {
    const res = await fetch("https://api.opendota.com/api/heroes", { next: { revalidate: 86400 } });
    return res.json();
  } catch { return []; }
}

async function fetchItemNames(): Promise<Map<number, { name: string; dname: string }>> {
  try {
    const res = await fetch("https://api.opendota.com/api/constants/items", { next: { revalidate: 86400 } });
    const items = await res.json();
    const map = new Map<number, { name: string; dname: string }>();
    for (const [name, item] of Object.entries(items) as [string, any][]) {
      if (item.id && item.dname) map.set(item.id, { name, dname: item.dname });
    }
    return map;
  } catch { return new Map(); }
}

function loadDraftData(): DraftData | null {
  const p = join(process.cwd(), "public", "draft-data.json");
  if (!existsSync(p)) return null;
  return JSON.parse(readFileSync(p, "utf-8"));
}

export default async function Page() {
  const [heroes, itemNames] = await Promise.all([fetchHeroes(), fetchItemNames()]);
  const draftData = loadDraftData();
  const itemNamesJson = Object.fromEntries(itemNames);

  return <DraftApp heroes={heroes} draftData={draftData} itemNames={itemNamesJson} />;
}
