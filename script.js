const submitButton = document.querySelector(".submit-button");
submitButton.addEventListener("click", getFormValues);
function getFormValues() {
  const tempValueElement = document.querySelector(".temp-input");
  const tempUnitElement = document.querySelector(".unit-input");

  const resultDiv = document.querySelector(".result");
  const firstConversion = document.querySelector(".first-conversion");
  const secondConversion = document.querySelector(".second-conversion");
  firstConversion.textContent = "";
  secondConversion.textContent = "";

  const tempValue = Number(tempValueElement.value);
  const tempUnit = tempUnitElement.value;

  if (isNaN(tempValue)) {
    firstConversion.textContent = "Please enter a valid temperature.";
    return;
  }

  var firstTemp = 0;
  var secondTemp = 0;
  if (tempUnit == "kelvin") {
    firstTemp = tempValue - 273.15;
    secondTemp = (9 / 5) * firstTemp + 32;
    firstConversion.textContent = `${tempValue.toFixed(2)}° Kelvin = ${firstTemp.toFixed(2)}° Celsius`;
    secondConversion.textContent = `${tempValue.toFixed(2)}° Kelvin = ${secondTemp.toFixed(2)}° Fahrenheit`;
  } else if (tempUnit == "celsius") {
    firstTemp = tempValue + 273.15;
    secondTemp = (9 / 5) * tempValue + 32;
    firstConversion.textContent = `${tempValue.toFixed(2)}° Celsius = ${firstTemp.toFixed(2)}° Kelvin `;
    secondConversion.textContent = `${tempValue.toFixed(2)}° Celsius = ${secondTemp.toFixed(2)}° Fahrenheit`;
  } else if (tempUnit == "fahrenheit") {
    firstTemp = (5 / 9) * (tempValue - 32);
    secondTemp = firstTemp + 273.15;
    firstConversion.textContent = `${tempValue.toFixed(2)}° Fahrenheit = ${firstTemp.toFixed(2)}° Celsius`;
    secondConversion.textContent = `${tempValue.toFixed(2)}° Fahrenheit = ${secondTemp.toFixed(2)}° Kelvin`;
  } else {
    firstConversion.textContent = "Please select a valid unit.";
  }

  resultDiv.appendChild(firstConversion);
  resultDiv.appendChild(secondConversion);
}
