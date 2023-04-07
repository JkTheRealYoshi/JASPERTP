const calculateBtn = document.getElementById("calculate-btn");
const assetInput = document.getElementById("asset");
const entryInput = document.getElementById("entry");
const tpInput = document.getElementById("tp");
const slInput = document.getElementById("sl");
const optionSelect = document.getElementById("option");
const probabilityOutput = document.getElementById("probability");
const suggestedTpOutput = document.getElementById("suggested-tp");
const resultStatusOutput = document.getElementById("result-status");
const historyList = document.getElementById("history-list");
const timeframeSelect = document.getElementById("timeframe-select");
const chartContainer = document.getElementById("chart-container");

calculateBtn.addEventListener("click", async () => {
  try {
    const response = await axios.post("/predict", {
      asset: assetInput.value,
      entry: parseFloat(entryInput.value),
      tp: parseFloat(tpInput.value),
      sl: parseFloat(slInput.value),
      option: optionSelect.value,
    });

    const data = response.data;
    probabilityOutput.textContent = `Probability: ${data.probability}%`;
    suggestedTpOutput.textContent = `Suggested TP: ${data.tp_suggest}`;
    resultStatusOutput.textContent = "Success";
    resultStatusOutput.classList.add("success");
    resultStatusOutput.classList.remove("failure");

    const li = document.createElement("li");
    li.innerHTML = `
      <span>Asset: ${assetInput.value}</span>
      <span>Entry: ${entryInput.value}</span>
      <span>TP: ${tpInput.value}</span>
      <span>SL: ${slInput.value}</span>
      <span>Option: ${optionSelect.value}</span>
      <span>Probability: ${data.probability}%</span>
    `;
    historyList.appendChild(li);

   updateChart(data);
  } catch (error) {
    resultStatusOutput.textContent = "Failure";
    resultStatusOutput.classList.add("failure");
    resultStatusOutput.classList.remove("success");
    console.error(error);
  }
});

timeframeSelect.addEventListener("change", async () => {
  try {
    const response = await axios.post("/get_chart", {
      asset: assetInput.value,
      timeframe: timeframeSelect.value,
    });

    const data = response.data;
    updateChart(data);
  } catch (error) {
    console.error(error);
  }
});

function updateChart(data) {
  chartContainer.innerHTML = "";
  const longChart = document.createElement("img");
  longChart.src = "data:image/png;base64," + data.long_prediction.chart;
  longChart.style.width = "100%";
  longChart.style.height = "auto";
  chartContainer.appendChild(longChart);

  const shortChart = document.createElement("img");
  shortChart.src = "data:image/png;base64," + data.short_prediction.chart;
  shortChart.style.width = "100%";
  shortChart.style.height = "auto";
  chartContainer.appendChild(shortChart);
}
