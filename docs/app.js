const data = window.FOURTH_AND_DATA;

// GitHub Pages URLs contain both the account and repository name, so the
// correct repo link can be filled automatically after the site is published.
if (location.hostname.endsWith("github.io")) {
  const account = location.hostname.split(".")[0];
  const repository = location.pathname.split("/").filter(Boolean)[0];
  if (account && repository) {
    document.querySelector("#repo-link").href = `https://github.com/${account}/${repository}`;
  }
}

const format = (value, suffix = "%") => `${Number(value).toFixed(2)}${suffix}`;
const money = value => new Intl.NumberFormat("en-AU", {
  style: "currency",
  currency: "AUD",
  maximumFractionDigits: 0,
}).format(Number(value || 0));

const seasonMetricValue = metric => {
  if (metric.format === "percent") return format(metric.value);
  if (metric.format === "money") return money(metric.value);
  return Number(metric.value || 0).toLocaleString("en-AU");
};

const animateBars = () => {
  requestAnimationFrame(() => {
    document.querySelectorAll(".bar-fill").forEach(bar => {
      bar.style.width = bar.dataset.width;
    });
  });
};

document.querySelector("#metric-grid").innerHTML = data.metrics.map((metric, index) => `
  <article class="metric ${index === 2 ? "highlight" : ""}">
    <span class="label">${metric.label}</span>
    <strong class="value">${format(metric.value, metric.suffix)}</strong>
    <span class="detail">${metric.detail}</span>
  </article>
`).join("");

const chart = document.querySelector("#strategy-chart");
if (data.strategies.length) {
  const min = Math.min(...data.strategies.map(item => item.points)) - 1;
  const max = Math.max(...data.strategies.map(item => item.points)) + 0.2;
  const best = Math.max(...data.strategies.map(item => item.points));
  chart.innerHTML = data.strategies.map(strategy => {
    const width = Math.max(8, ((strategy.points - min) / (max - min)) * 100);
    return `
      <div class="bar-row ${strategy.points === best ? "best" : ""}">
        <span>${strategy.name}</span>
        <div class="bar-track"><div class="bar-fill" data-width="${width}%"></div></div>
        <strong class="bar-value">${format(strategy.points)}</strong>
      </div>`;
  }).join("");
} else {
  chart.innerHTML = `<div class="chart-empty"><strong>Awaiting backtest data</strong><span>Run the model summary workflow to populate this comparison.</span></div>`;
}

document.querySelector("#method-steps").innerHTML = data.method.map((step, index) => `
  <article class="step">
    <span class="step-num">0${index + 1}</span>
    <h3>${step.title}</h3>
    <p>${step.text}</p>
  </article>
`).join("");

const picksBody = document.querySelector("#picks-body");
if (data.picks.length) {
  document.querySelector("#week-label").textContent = data.week;
  document.querySelector("#picks-status").textContent = "MODEL READY";
  document.querySelector("#empty-note").hidden = true;
  picksBody.innerHTML = data.picks.map(pick => `
    <tr>
      <td>${pick.rank}</td>
      <td><strong>${pick.pick}</strong></td>
      <td>${pick.opponent}</td>
      <td>${format(pick.probability)}</td>
      <td>${pick.decimal_price ? Number(pick.decimal_price).toFixed(2) : "—"}</td>
      <td>${pick.stake ? money(pick.stake) : "—"}</td>
      <td>${pick.potential_return ? money(pick.potential_return) : "—"}</td>
      <td><span class="bet-status ${String(pick.status).toLowerCase()}">${pick.result || pick.status}</span></td>
    </tr>
  `).join("");
} else {
  picksBody.innerHTML = `<tr><td>—</td><td><strong>Schedule pending</strong></td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>`;
}

const season = data.confidence_pool || { season: null, metrics: [], weekly: [] };
const seasonLabel = document.querySelector("#season-label");
seasonLabel.textContent = season.season ? `${season.season} SEASON` : "SEASON TRACKER";

const seasonMetrics = document.querySelector("#season-metrics");
if (season.metrics.length) {
  seasonMetrics.innerHTML = season.metrics.map(metric => `
    <article class="betting-metric">
      <span>${metric.label}</span>
      <strong>${seasonMetricValue(metric)}</strong>
      <small>${metric.detail}</small>
    </article>
  `).join("");
} else {
  seasonMetrics.innerHTML = `
    <article class="betting-metric">
      <span>Season tracker</span>
      <strong>—</strong>
      <small>Record and settle the first confidence card to begin.</small>
    </article>`;
}

const seasonChart = document.querySelector("#season-chart");
const seasonSummaryCopy = document.querySelector("#season-summary-copy");
const seasonLatestWeek = document.querySelector("#season-latest-week");

if (!season.weekly.length) {
  seasonChart.innerHTML = `<div class="chart-empty"><strong>Awaiting settled confidence picks</strong><span>Week-by-week points performance will appear here.</span></div>`;
  seasonSummaryCopy.textContent = "The live ledger will track every straight-up pick, its assigned confidence weight and the points earned once the result is final.";
  seasonLatestWeek.textContent = "No settled week yet";
} else {
  const bestWeek = Math.max(...season.weekly.map(week => Number(week.points_percent)));
  seasonChart.innerHTML = season.weekly.map(week => `
    <div class="bar-row ${Number(week.points_percent) === bestWeek ? "best" : ""}">
      <span>Week ${week.week_number}</span>
      <div class="bar-track"><div class="bar-fill" data-width="${Math.max(8, Number(week.points_percent))}%"></div></div>
      <strong class="bar-value">${format(week.points_percent)}</strong>
    </div>
  `).join("");

  const latest = season.weekly.at(-1);
  const totals = season.metrics.reduce((lookup, metric) => {
    lookup[metric.label] = metric;
    return lookup;
  }, {});
  const accuracy = totals["Pick accuracy"]?.value ?? 0;
  const points = totals["Confidence points"]?.value ?? 0;
  const pointsPercent = totals["Points percentage"]?.value ?? 0;

  seasonSummaryCopy.textContent = `Through ${season.weekly.length} settled week${season.weekly.length === 1 ? "" : "s"}, the card is hitting ${format(accuracy)} of picks and has banked ${Number(points).toLocaleString("en-AU")} confidence points at ${format(pointsPercent)} efficiency.`;
  seasonLatestWeek.textContent = `${latest.week}: ${latest.confidence_points}/${latest.max_points} points · ${format(latest.points_percent)}`;
}

const betting = data.betting || { metrics: [], weekly: [] };
document.querySelector("#betting-metrics").innerHTML = betting.metrics.map(metric => `
  <article class="betting-metric">
    <span>${metric.label}</span>
    <strong>${metric.format === "money" ? money(metric.value) : format(metric.value)}</strong>
    <small>${metric.detail}</small>
  </article>
`).join("");

const profitChart = document.querySelector("#profit-chart");
if (!betting.weekly.length) {
  profitChart.innerHTML = `<div class="chart-empty"><strong>Awaiting settled games</strong><span>The P/L line will begin after the first card is settled.</span></div>`;
} else {
  const values = [0, ...betting.weekly.map(week => Number(week.cumulative_profit))];
  const width = 700;
  const height = 260;
  const pad = 34;
  const minimum = Math.min(...values, 0);
  const maximum = Math.max(...values, 0);
  const range = maximum - minimum || 1;
  const x = index => pad + index * ((width - pad * 2) / (values.length - 1 || 1));
  const y = value => pad + (maximum - value) * ((height - pad * 2) / range);
  const points = values.map((value, index) => `${x(index)},${y(value)}`).join(" ");
  const zeroY = y(0);

  profitChart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" aria-hidden="true">
      <line class="zero-line" x1="${pad}" y1="${zeroY}" x2="${width - pad}" y2="${zeroY}" />
      <polyline class="profit-line" points="${points}" />
      ${values.map((value, index) => `<circle cx="${x(index)}" cy="${y(value)}" r="4" />`).join("")}
      <text x="${pad}" y="${height - 5}">START</text>
      <text text-anchor="end" x="${width - pad}" y="${height - 5}">${betting.weekly.at(-1).week}</text>
      <text x="${pad}" y="18">${money(maximum)}</text>
      <text x="${pad}" y="${height - 14}">${money(minimum)}</text>
    </svg>`;
}

animateBars();
document.querySelector("#year").textContent = new Date().getFullYear();
