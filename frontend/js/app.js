// ==========================================================
// CONFIGURATION
// ==========================================================

// Approximate USD → INR conversion rate.
//
// This is a display conversion only.
// The ML model always predicts salary in USD.
const USD_TO_INR = 88.0;

// Store the latest prediction in USD.
let latestSalaryUSD = null;

// API is accessed through the frontend's Nginx reverse proxy.
const API_URL = "/api/v1/predict";

const form = document.getElementById("salary-form");

const button = document.getElementById("predict-button");

const loading = document.getElementById("loading");

const errorBox = document.getElementById("error");

const result = document.getElementById("result");

const salaryValue = document.getElementById("salary-value");

const modelInfo = document.getElementById("model-info");

const currencySelector = document.getElementById("currency");

const conversionInfo = document.getElementById("conversion-info");

// ==========================================================
// FORM SUBMISSION
// ==========================================================

form.addEventListener("submit", async function (event) {
  event.preventDefault();

  hideError();
  hideResult();

  button.disabled = true;

  loading.classList.remove("hidden");

  try {
    const payload = {
      title: document.getElementById("title").value.trim().toLowerCase(),

      skill_list: document
        .getElementById("skill_list")
        .value.trim()
        .toLowerCase(),

      formatted_experience_level: document
        .getElementById("experience")
        .value.trim()
        .toLowerCase(),

      company_state: getOptionalValue("state"),

      company_country: getOptionalValue("country"),

      top_industry: getOptionalValue("industry"),
    };

    console.log("Sending prediction request:", payload);

    const response = await fetch(API_URL, {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Prediction request failed.");
    }

    displayResult(data);
  } catch (error) {
    console.error("Prediction error:", error);

    showError(error.message || "Unable to get prediction.");
  } finally {
    button.disabled = false;

    loading.classList.add("hidden");
  }
});

currencySelector.addEventListener("change", function () {
  updateSalaryDisplay();
});

// ==========================================================
// OPTIONAL FIELD HELPER
// ==========================================================

function getOptionalValue(id) {
  const value = document.getElementById(id).value.trim();

  return value === "" ? null : value;
}

// ==========================================================
// DISPLAY RESULT
// ==========================================================

function displayResult(data) {
  // ------------------------------------------------------
  // Store the canonical model output.
  // The model prediction is always USD.
  // ------------------------------------------------------

  latestSalaryUSD = Number(data.predicted_annual_salary);

  if (!Number.isFinite(latestSalaryUSD)) {
    throw new Error("Invalid salary returned by the API.");
  }

  // Reset currency to USD whenever
  // a new prediction is generated.

  currencySelector.value = "USD";

  // Display salary.

  updateSalaryDisplay();

  // Model information.

  modelInfo.textContent =
    `Model: ${data.model_name} | ` + `Version alias: ${data.model_alias}`;

  result.classList.remove("hidden");
}

// ==========================================================
// CURRENCY DISPLAY
// ==========================================================

function updateSalaryDisplay() {
  if (latestSalaryUSD === null) {
    return;
  }

  const currency = currencySelector.value;

  // ------------------------------------------------------
  // USD
  // ------------------------------------------------------

  if (currency === "USD") {
    salaryValue.textContent = formatCurrency(latestSalaryUSD, "USD", "en-US");

    conversionInfo.classList.add("hidden");

    return;
  }

  // ------------------------------------------------------
  // INR
  // ------------------------------------------------------

  if (currency === "INR") {
    const salaryINR = latestSalaryUSD * USD_TO_INR;

    salaryValue.textContent = formatCurrency(salaryINR, "INR", "en-IN");

    conversionInfo.textContent =
      `Approx. conversion: ` + `1 USD = ₹${USD_TO_INR}`;

    conversionInfo.classList.remove("hidden");
  }
}

function formatCurrency(value, currency, locale) {
  return new Intl.NumberFormat(locale, {
    style: "currency",

    currency: currency,

    maximumFractionDigits: 0,
  }).format(value);
}

// ==========================================================
// ERROR
// ==========================================================

function showError(message) {
  errorBox.textContent = message;

  errorBox.classList.remove("hidden");
}

function hideError() {
  errorBox.classList.add("hidden");

  errorBox.textContent = "";
}

// ==========================================================
// RESULT
// ==========================================================

function hideResult() {
  result.classList.add("hidden");
}
