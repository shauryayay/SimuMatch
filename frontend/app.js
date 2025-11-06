document.getElementById("btnMatch").onclick = async () => {
  const r = await fetch("/match", { method: "POST" });
  const data = await r.json();
  const container = document.getElementById("summary");
  container.innerHTML = "<pre>" + JSON.stringify(data, null, 2) + "</pre>";
};

document.getElementById("btnEvents").onclick = async () => {
  const loc = document.getElementById("loc").value;
  const radius = parseInt(document.getElementById("radius").value || "50", 10);

  const body = loc ? {location: loc, radius_km: radius} : {location: null, radius_km: radius};
  const r = await fetch("/recommend_events", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  const data = await r.json();
  const el = document.getElementById("events");
  el.innerHTML = "";
  data.events.slice(0, 20).forEach(ev => {
    const div = document.createElement("div");
    div.className = "ev";
    div.innerHTML = `<b>${ev.name || "untitled"}</b><br/>${ev.start} — ${ev.city || ""} <br/><a href="${ev.url}" target="_blank">register</a>`;
    el.appendChild(div);
  });
};
