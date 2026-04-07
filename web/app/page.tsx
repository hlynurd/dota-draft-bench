import type { OpenDotaHero } from "./components/HeroPicker";
import DraftApp from "./components/DraftApp";

async function fetchHeroes(): Promise<OpenDotaHero[]> {
  try {
    const res = await fetch("https://api.opendota.com/api/heroes", { next: { revalidate: 86400 } });
    return res.json();
  } catch { return []; }
}

async function fetchItemNames(): Promise<Record<number, { name: string; dname: string }>> {
  try {
    const res = await fetch("https://api.opendota.com/api/constants/items", { next: { revalidate: 86400 } });
    const items = await res.json();
    const map: Record<number, { name: string; dname: string }> = {};
    for (const [name, item] of Object.entries(items) as [string, any][]) {
      if (item.id && item.dname) map[item.id] = { name, dname: item.dname };
    }
    return map;
  } catch { return {}; }
}

export default async function Page() {
  const [heroes, itemNames] = await Promise.all([fetchHeroes(), fetchItemNames()]);

  return <DraftApp heroes={heroes} draftData={null} itemNames={itemNames} />;
}
