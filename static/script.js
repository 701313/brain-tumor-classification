const form = document.querySelector("#uploadForm");
const imageInput = document.querySelector("#imageInput");
const dropzone = document.querySelector("#dropzone");
const previewWrap = document.querySelector("#previewWrap");
const previewImage = document.querySelector("#previewImage");
const thresholdInput = document.querySelector("#thresholdInput");
const thresholdValue = document.querySelector("#thresholdValue");
const submitBtn = document.querySelector("#submitBtn");
const emptyState = document.querySelector("#emptyState");
const resultContent = document.querySelector("#resultContent");
const resultImage = document.querySelector("#resultImage");
const tumorType = document.querySelector("#tumorType");
const confidenceScore = document.querySelector("#confidenceScore");
const confidenceBar = document.querySelector("#confidenceBar");
const downloadImage = document.querySelector("#downloadImage");
const downloadReport = document.querySelector("#downloadReport");
const modelStatus = document.querySelector("#modelStatus");
const inferenceMode = document.querySelector("#inferenceMode");
const toast = document.querySelector("#toast");
const themeToggle = document.querySelector("#themeToggle");
const languageSelect = document.querySelector("#languageSelect");
const accuracyValue = document.querySelector("#accuracyValue");

const translations = {
  en: {
    subtitle: "MRI tumor detection",
    classes: "Classes",
    imageSize: "Image size",
    validation: "Validation accuracy",
    language: "Language",
    darkMode: "Dark mode",
    title: "Brain Tumor Detection",
    description:
      "Upload an MRI image to localize and classify glioma, meningioma, pituitary tumor, or no tumor.",
    uploadMri: "MRI Input",
    dropTitle: "Drop MRI image here",
    dropHint: "or click to browse",
    threshold: "Confidence threshold",
    analyze: "Analyze MRI",
    result: "Detection Result",
    emptyTitle: "No scan analyzed yet",
    emptyText: "Results and annotated image will appear here.",
    tumorType: "Tumor type",
    confidence: "Confidence",
    downloadImage: "Download image",
    downloadReport: "Download report",
    pipeline: "Processing Pipeline",
    resize: "Resize",
    denoise: "Denoise",
    normalize: "Normalize",
    detect: "Detect",
  },
  hi: {
    subtitle: "MRI ट्यूमर पहचान",
    classes: "वर्ग",
    imageSize: "छवि आकार",
    validation: "मान्यता सटीकता",
    language: "भाषा",
    darkMode: "डार्क मोड",
    title: "ब्रेन ट्यूमर डिटेक्शन",
    description:
      "ग्लियोमा, मेनिंजियोमा, पिट्यूटरी ट्यूमर या नो ट्यूमर को पहचानने के लिए MRI छवि अपलोड करें।",
    uploadMri: "MRI इनपुट",
    dropTitle: "MRI छवि यहां छोड़ें",
    dropHint: "या ब्राउज करने के लिए क्लिक करें",
    threshold: "कॉन्फिडेंस थ्रेशोल्ड",
    analyze: "MRI विश्लेषण करें",
    result: "डिटेक्शन परिणाम",
    emptyTitle: "अभी कोई स्कैन विश्लेषित नहीं",
    emptyText: "परिणाम और एनोटेटेड छवि यहां दिखेंगे।",
    tumorType: "ट्यूमर प्रकार",
    confidence: "कॉन्फिडेंस",
    downloadImage: "छवि डाउनलोड करें",
    downloadReport: "रिपोर्ट डाउनलोड करें",
    pipeline: "प्रोसेसिंग पाइपलाइन",
    resize: "रीसाइज",
    denoise: "डिनॉइज",
    normalize: "नॉर्मलाइज",
    detect: "डिटेक्ट",
  },
  es: {
    subtitle: "Detección tumoral MRI",
    classes: "Clases",
    imageSize: "Tamaño",
    validation: "Precisión validación",
    language: "Idioma",
    darkMode: "Modo oscuro",
    title: "Detección de Tumor Cerebral",
    description:
      "Sube una MRI para localizar y clasificar glioma, meningioma, tumor pituitario o sin tumor.",
    uploadMri: "Entrada MRI",
    dropTitle: "Suelta la imagen MRI",
    dropHint: "o haz clic para buscar",
    threshold: "Umbral de confianza",
    analyze: "Analizar MRI",
    result: "Resultado",
    emptyTitle: "Aún no hay análisis",
    emptyText: "Los resultados y la imagen anotada aparecerán aquí.",
    tumorType: "Tipo de tumor",
    confidence: "Confianza",
    downloadImage: "Descargar imagen",
    downloadReport: "Descargar reporte",
    pipeline: "Pipeline",
    resize: "Redimensionar",
    denoise: "Eliminar ruido",
    normalize: "Normalizar",
    detect: "Detectar",
  },
};

thresholdInput.addEventListener("input", () => {
  thresholdValue.textContent = `${thresholdInput.value}%`;
});

imageInput.addEventListener("change", () => previewSelectedImage(imageInput.files[0]));

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  const file = event.dataTransfer.files[0];
  if (!file) return;
  imageInput.files = event.dataTransfer.files;
  previewSelectedImage(file);
});

themeToggle.addEventListener("change", () => {
  document.body.classList.toggle("dark", themeToggle.checked);
  localStorage.setItem("theme", themeToggle.checked ? "dark" : "light");
});

languageSelect.addEventListener("change", () => applyLanguage(languageSelect.value));

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = imageInput.files[0];
  if (!file) {
    showToast("Please select an MRI image first.");
    return;
  }

  const payload = new FormData();
  payload.append("image", file);
  payload.append("threshold", String(Number(thresholdInput.value) / 100));

  submitBtn.disabled = true;
  modelStatus.textContent = "Analyzing";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: payload,
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Prediction failed.");
    }
    renderResult(data.result);
  } catch (error) {
    showToast(error.message);
    modelStatus.textContent = "Ready";
  } finally {
    submitBtn.disabled = false;
  }
});

function previewSelectedImage(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    previewImage.src = reader.result;
    previewWrap.hidden = false;
  };
  reader.readAsDataURL(file);
}

function renderResult(result) {
  emptyState.hidden = true;
  resultContent.hidden = false;

  resultImage.src = `${result.annotated_image_url}?t=${Date.now()}`;
  tumorType.textContent = result.tumor_type;
  confidenceScore.textContent = `${result.confidence_percent}%`;
  confidenceBar.value = result.confidence_percent;
  downloadImage.href = result.annotated_image_url;
  downloadReport.href = result.report_url;
  modelStatus.textContent = result.model_status;
  inferenceMode.textContent = result.inference_mode;
  accuracyValue.textContent = `${result.accuracy_display.value}%`;

  document.querySelector("#resizeStep").textContent = result.preprocessing.resize;
  document.querySelector("#denoiseStep").textContent = result.preprocessing.denoising;
  document.querySelector("#normalizeStep").textContent = result.preprocessing.normalization;
  document.querySelector("#clinicalNote").textContent = result.clinical_note;
}

function applyLanguage(language) {
  const dictionary = translations[language] || translations.en;
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const key = node.dataset.i18n;
    if (dictionary[key]) node.textContent = dictionary[key];
  });
  localStorage.setItem("language", language);
}

function showToast(message) {
  toast.textContent = message;
  toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    toast.hidden = true;
  }, 4200);
}

function bootstrapPreferences() {
  const savedTheme = localStorage.getItem("theme");
  themeToggle.checked = savedTheme === "dark";
  document.body.classList.toggle("dark", themeToggle.checked);

  const savedLanguage = localStorage.getItem("language") || "en";
  languageSelect.value = savedLanguage;
  applyLanguage(savedLanguage);
}

bootstrapPreferences();
