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

document.querySelector("#metric-grid").innerHTML = data.metrics.map((metric, index) => `
  <article class="metric ${index === 2 ? "highlight" : ""}">
    <span class="label">${metric.label}</span>
    <strong class="value">${format(metric.value, metric.suffix)}</strong>
    <span class="detail">${metric.detail}</span>
  </article>
`).join("");

const chart = document.querySelector("#strategy-chart");
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

requestAnimationFrame(() => {
  document.querySelectorAll(".bar-fill").forEach(bar => { bar.style.width = bar.dataset.width; });
});

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
      <td><div class="confidence-bar" aria-label="Confidence ${pick.confidence}"><i style="width:${pick.confidence}%"></i></div></td>
    </tr>
  `).join("");
} else {
  picksBody.innerHTML = `<tr><td>—</td><td><strong>Schedule pending</strong></td><td>—</td><td>—</td><td>—</td></tr>`;
}

document.querySelector("#year").textContent = new Date().getFullYear();
