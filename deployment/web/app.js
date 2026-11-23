const $ = (id) => document.getElementById(id);
const input = $("file");
const dropzone = $("dropzone");
const inspect = $("inspect");
const sample = $("sample");
const preview = $("input-preview");
const builtin = { id: "pcb1", name: "PCB1 · HC-SR04", builtin: true, versions: [] };
const groups = {
  train_normal: "Train · Normal",
  val_normal: "Validation · Normal",
  val_anomaly: "Validation · Defective",
  test_normal: "Test · Normal",
  test_anomaly: "Test · Defective",
};
const decisions = {
  normal: "Normal appearance",
  anomaly: "Possible defect",
  cannot_inspect: "Unable to inspect",
};
const failureReasons = {
  missing_image: "The image file is missing.",
  invalid_image: "Choose a valid JPEG or PNG image.",
  multiple_frames: "Choose a single-frame image.",
  no_product_detected: "No HC-SR04 module was detected.",
  invalid_detection: "The detected region could not be cropped.",
};
const imageLabels = {
  crop: "Product crop",
  detected: "Full image with crop boundary",
  mask: "Binary defect mask",
};

let workspace = { products: [builtin], cloud: { available: false } };
let workspaceLoaded = false;
let refreshPromise;
let currentPage = "inspection";
let chosenFile = null;
let objectUrl = null;
let currentImages = {};
let busy = false;
let datasetBusy = false;
let trainingBusy = false;

async function api(path, { method = "GET", body } = {}) {
  const upload = body instanceof Blob;
  const response = await fetch(path, {
    method,
    headers: body === undefined ? undefined : {
      "Content-Type": upload ? "application/octet-stream" : "application/json",
    },
    body: body === undefined || upload ? body : JSON.stringify(body),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok || data === null) {
    throw new Error(data?.error || data?.message || "The application could not complete the request.");
  }
  return data;
}

function showNotice(id, message = "", error = false) {
  const notice = $(id);
  notice.textContent = message;
  notice.classList.toggle("error", error);
  notice.hidden = !message;
}

function errorText(error) {
  if (error instanceof TypeError) {
    return "Connection lost. Check that the application is running and try again.";
  }
  return error.message;
}

function selectedProduct(select) {
  return workspace.products.find(({ id }) => id === $(select).value);
}

function isBuiltin(product) {
  return Boolean(product?.builtin || product?.id === "pcb1");
}

function setOptions(select, options, fallback = "") {
  const target = $(select);
  const previous = target.value;
  const current = Array.from(target.options, (option) => [option.value, option.textContent]);
  if (JSON.stringify(current) !== JSON.stringify(options)) {
    target.replaceChildren(...options.map(([value, name]) => new Option(name, value)));
    const preferred = options.some(([value]) => value === previous) ? previous : fallback;
    if (options.some(([value]) => value === preferred)) target.value = preferred;
  }
}

function renderSelectors() {
  const products = workspace.products.map(({ id, name }) => [id, name]);
  const custom = workspace.products.filter((product) => !isBuiltin(product));
  const managed = custom.map(({ id, name }) => [id, name]);

  setOptions("inspection-product", products, "pcb1");
  setOptions("dataset-product", managed.length ? managed : [["", "No products yet"]]);
  setOptions("training-product", managed.length ? managed : [["", "No products yet"]]);
  $("dataset-product").disabled = datasetBusy || !managed.length;
  $("training-product").disabled = !managed.length;
}

function showInspectionSection(id) {
  for (const section of ["empty", "loading", "result"]) {
    $(section).hidden = section !== id;
  }
}

function hasInspectionModel(product) {
  const version = $("inspection-version").value;
  return product?.available !== false && (isBuiltin(product) || Boolean(version || product?.active_version));
}

function syncInspectionControls() {
  const product = selectedProduct("inspection-product");
  const available = hasInspectionModel(product);
  inspect.disabled = busy || !chosenFile || !available;
  sample.hidden = !isBuiltin(product);
  sample.disabled = busy || product?.available === false;
  input.disabled = busy;
  $("inspection-product").disabled = busy;
  $("inspection-version").disabled = busy || isBuiltin(product) || !product?.versions?.length;
}

function renderInspectionModel() {
  const product = selectedProduct("inspection-product");
  const active = product?.active_version || "";
  const options = isBuiltin(product) && product.available !== false ? [[active, "PCB1 / v1"]] : (product?.versions || []).map((version) => [
    version.id, (version.name || version.id) + (version.id === active ? " · Active" : ""),
  ]);
  if (!isBuiltin(product) && options.length && !active) options.unshift(["", "Choose a version"]);
  setOptions(
    "inspection-version",
    options.length ? options : [["", "No trained versions"]],
    active,
  );
  let note = "Train a model or choose an existing version to inspect this product.";
  if (isBuiltin(product)) {
    note = product.available === false
      ? "Add the PCB1 model weights to the project to inspect this product."
      : "PCB1 locates the board before evaluating the crop.";
  } else if (hasInspectionModel(product)) {
    note = "Use the same framing as the training images. The model inspects the full frame.";
  }
  $("inspection-model-note").textContent = note;
  syncInspectionControls();
}

function showPage() {
  const requested = location.hash.slice(1);
  currentPage = ["inspection", "products", "training"].includes(requested) ? requested : "inspection";
  for (const link of document.querySelectorAll(".nav-link")) {
    const selected = link.dataset.page === currentPage;
    if (selected) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
    $("page-" + link.dataset.page).hidden = !selected;
  }
  if (currentPage === "training" && workspaceLoaded) refreshWorkspace();
}

function renderDataset() {
  const product = selectedProduct("dataset-product");
  $("dataset-content").hidden = !product;
  $("dataset-empty").hidden = Boolean(product);
  if (!product) return;

  $("dataset-counts").replaceChildren(...Object.entries(groups).map(([id, label]) => {
    const cell = document.createElement("div");
    const title = document.createElement("dt");
    const count = document.createElement("dd");
    title.textContent = label;
    count.textContent = product.dataset?.[id] ?? 0;
    cell.append(title, count);
    return cell;
  }));
}

function setRowStatus(row, status, label) {
  const chip = document.createElement("span");
  chip.className = "row-status";
  chip.dataset.status = status;
  chip.textContent = label;
  row.append(chip);
}

function listMessage(list, message) {
  const note = document.createElement("p");
  note.className = "help";
  note.textContent = message;
  list.append(note);
}

function modelRow(title, details) {
  const row = document.createElement("div");
  row.className = "item-row";
  const text = document.createElement("div");
  const name = document.createElement("strong");
  name.textContent = title;
  text.append(name);
  for (const value of details.filter(Boolean)) {
    const line = document.createElement("p");
    line.textContent = value;
    text.append(line);
  }
  row.append(text);
  return row;
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function evaluationLine(label, evaluation) {
  const details = [];
  if (Number.isFinite(evaluation.balanced_accuracy)) {
    const accuracy = (evaluation.balanced_accuracy * 100).toFixed(1);
    details.push(accuracy + "% balanced accuracy");
  }
  if (Number.isFinite(evaluation.auroc)) details.push(evaluation.auroc.toFixed(3) + " AUROC");
  if (Number.isInteger(evaluation.false_positive)) details.push(evaluation.false_positive + " false positives");
  if (Number.isInteger(evaluation.false_negative)) details.push(evaluation.false_negative + " false negatives");
  const size = evaluation.normal + evaluation.anomaly;
  const heading = Number.isInteger(size) ? label + " (" + size + " images)" : label;
  return details.length ? [heading, ...details].join(" · ") : heading;
}

function metricLines(metrics) {
  if (!metrics) return [];
  const details = [];
  if (Number.isInteger(metrics.training_images)) details.push(metrics.training_images + " normal training images");
  if (metrics.device) details.push(metrics.device.startsWith("cuda") ? "GPU" : "CPU");
  const lines = [details.join(" · ")];
  if (metrics.validation) lines.push(evaluationLine("Validation", metrics.validation));
  lines.push(metrics.test ? evaluationLine("Test", metrics.test) : "No held-out test images");
  return lines;
}

function renderTraining() {
  const product = selectedProduct("training-product");
  const provider = $("training-provider");
  const cloud = workspace.cloud || {};
  const cloudReady = Boolean(cloud.available);
  provider.options[0].disabled = !cloudReady;
  if (!cloudReady && provider.value === "modal") provider.value = "local";
  $("training-provider-note").textContent = cloudReady
    ? "Modal uses a cloud GPU. Local CPU runs on this computer."
    : cloud.reason || "Modal is not configured. Local CPU training is available.";
  $("service-label").textContent = cloudReady ? "Modal GPU configured" : "Local CPU available";
  const jobs = product?.jobs || [];
  $("training-submit").disabled = trainingBusy || !product ||
    jobs.some((job) => ["queued", "running"].includes(job.status));

  const jobList = $("job-list");
  jobList.replaceChildren();
  if (!jobs.length) listMessage(jobList, product ? "No training runs for this product yet." : "Create a product and upload its dataset to start training.");
  for (const job of jobs) {
    const source = job.provider === "modal" ? "Modal GPU" : job.provider === "local" ? "Local CPU" : "Training";
    const name = job.name || [source, job.id.slice(0, 8)].join(" · ");
    const row = modelRow(name, [formatDate(job.created_at), job.message]);
    const status = job.status || "unknown";
    setRowStatus(row, status, status[0].toUpperCase() + status.slice(1));
    jobList.append(row);
  }

  const versions = $("version-list");
  versions.replaceChildren();
  const models = product?.versions || [];
  if (!models.length) listMessage(versions, "A completed training run will add a model version here.");
  for (const version of models) {
    const row = modelRow(version.name || version.id, [formatDate(version.created_at), ...metricLines(version.metrics)]);
    if (product.active_version === version.id) {
      setRowStatus(row, "active", "Active");
    } else {
      const activate = document.createElement("button");
      activate.type = "button";
      activate.className = "button secondary";
      activate.textContent = "Activate";
      activate.addEventListener("click", () => activateVersion(product.id, version, activate));
      row.append(activate);
    }
    versions.append(row);
  }
}

function renderWorkspace() {
  renderSelectors();
  renderInspectionModel();
  renderDataset();
  renderTraining();
}

function refreshWorkspace(afterMutation = false) {
  if (refreshPromise) {
    return afterMutation ? refreshPromise.then(() => refreshWorkspace()) : refreshPromise;
  }
  refreshPromise = (async () => {
    try {
      const state = await api("/api/workspace");
      workspace = state;
      workspaceLoaded = true;
      showNotice("workspace-error");
      renderWorkspace();
    } catch (error) {
      if (!workspaceLoaded) $("service-label").textContent = "Workspace unavailable";
      showNotice("workspace-error", "Workspace unavailable. " + errorText(error), true);
    } finally {
      refreshPromise = null;
    }
  })();
  return refreshPromise;
}

function resetInspection() {
  currentImages = {};
  $("visual-image").removeAttribute("src");
  $("heatmap").removeAttribute("src");
  showNotice("input-error");
  showInspectionSection("empty");
}

function chooseImage(file) {
  if (!file.size || file.size > 20 * 1024 * 1024) {
    showNotice("input-error", "Choose a non-empty image up to 20 MB.", true);
    return;
  }
  if (objectUrl) URL.revokeObjectURL(objectUrl);

  chosenFile = file;
  objectUrl = URL.createObjectURL(file);
  preview.src = objectUrl;
  preview.hidden = false;
  $("change-photo").hidden = false;
  $("upload-content").hidden = true;
  $("filename").textContent = file.name;
  $("filename").hidden = false;
  resetInspection();
  syncInspectionControls();
}

function updateVisual() {
  const selected = $("view").value;
  $("visual-image").alt = imageLabels[selected];
  $("visual-image").src = currentImages[selected];
}

function renderResult(data) {
  $("verdict").dataset.status = data.status;
  $("decision").textContent = decisions[data.status] || "Inspection result unavailable";
  const detail = data.status === "cannot_inspect"
    ? failureReasons[data.reason] || "This image could not be inspected."
    : data.status === "anomaly" && data.mask_has_regions === false
      ? "The image score is elevated. No region exceeded the pixel threshold."
      : "";
  $("detail").textContent = detail;
  $("detail").hidden = !detail;
  const hasScore = Number.isFinite(data.anomaly_score);
  $("score").textContent = hasScore ? data.anomaly_score.toFixed(3) : "";
  $("score-group").hidden = !hasScore;

  currentImages = data.images || {};
  const availableViews = ["crop", "detected", "mask"].filter((key) => currentImages[key]);
  $("visuals").hidden = !availableViews.length || !currentImages.heatmap;
  if (!$("visuals").hidden) {
    const labels = { crop: "Product crop", detected: "Full image", mask: "Defect mask" };
    setOptions("view", availableViews.map((key) => [key, labels[key]]));
    $("view").value = availableViews[0];
    updateVisual();
    $("heatmap").src = currentImages.heatmap;
  }
  showInspectionSection("result");
}

preview.addEventListener("error", () => {
  preview.hidden = true;
  $("upload-content").hidden = false;
  $("change-photo").hidden = true;
});
input.addEventListener("change", () => {
  if (input.files[0]) chooseImage(input.files[0]);
  input.value = "";
});
for (const name of ["dragenter", "dragover"]) {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    if (!busy) dropzone.classList.add("dragging");
  });
}
dropzone.addEventListener("dragleave", (event) => {
  if (!dropzone.contains(event.relatedTarget)) dropzone.classList.remove("dragging");
});
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
  if (!busy && event.dataTransfer.files[0]) chooseImage(event.dataTransfer.files[0]);
});
$("inspection-product").addEventListener("change", () => {
  $("inspection-version").value = "";
  resetInspection();
  renderInspectionModel();
});
$("inspection-version").addEventListener("change", () => {
  resetInspection();
  syncInspectionControls();
});
$("view").addEventListener("change", updateVisual);

sample.addEventListener("click", async () => {
  busy = true;
  syncInspectionControls();
  showNotice("input-error");
  try {
    const response = await fetch("/sample.jpg");
    if (!response.ok) throw new Error("Sample image unavailable. Choose a local image instead.");
    const file = new File([await response.blob()], "PCB1-041.jpg", { type: "image/jpeg" });
    chooseImage(file);
  } catch (error) {
    showNotice("input-error", errorText(error), true);
  } finally {
    busy = false;
    syncInspectionControls();
  }
});

inspect.addEventListener("click", async () => {
  const product = selectedProduct("inspection-product");
  if (busy || !chosenFile || !hasInspectionModel(product)) return;
  busy = true;
  syncInspectionControls();
  showNotice("input-error");
  showInspectionSection("loading");
  try {
    const params = new URLSearchParams({ product_id: product.id });
    const version = $("inspection-version").value;
    if (version && !isBuiltin(product)) params.set("version_id", version);
    renderResult(await api("/inspect?" + params, { method: "POST", body: chosenFile }));
    if (window.matchMedia("(max-width: 640px)").matches) {
      $("result-panel").scrollIntoView({ block: "start" });
    }
  } catch (error) {
    showNotice("input-error", errorText(error), true);
    showInspectionSection("empty");
  } finally {
    busy = false;
    syncInspectionControls();
  }
});

$("create-product").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const submit = form.querySelector("button");
  submit.disabled = true;
  showNotice("product-message");
  try {
    const name = $("product-name").value.trim();
    if (!name) throw new Error("Enter a product name.");
    const data = await api("/api/products", { method: "POST", body: { name } });
    await refreshWorkspace(true);
    const created = data.product || data;
    if (created.id) {
      $("dataset-product").value = created.id;
      $("training-product").value = created.id;
      renderDataset();
      renderTraining();
    }
    form.reset();
    showNotice("product-message", name + " was created.");
  } catch (error) {
    showNotice("product-message", errorText(error), true);
  } finally {
    submit.disabled = false;
  }
});

$("dataset-product").addEventListener("change", () => {
  $("upload-dataset").reset();
  showNotice("dataset-message");
  renderDataset();
});
$("upload-dataset").addEventListener("submit", async (event) => {
  event.preventDefault();
  const product = selectedProduct("dataset-product");
  const group = $("dataset-group").value;
  const files = Array.from($("dataset-images").files);
  if (!product || !files.length || datasetBusy) return;
  datasetBusy = true;
  const form = event.currentTarget;
  const controls = [...form.elements, $("dataset-product")];
  controls.forEach((control) => { control.disabled = true; });
  let uploaded = 0;
  let existing = 0;
  let firstError = "";
  try {
    for (const [index, file] of files.entries()) {
      showNotice("dataset-message", "Uploading " + (index + 1) + " of " + files.length + "…");
      const params = new URLSearchParams({ group, filename: file.name });
      try {
        const result = await api("/api/products/" + encodeURIComponent(product.id) + "/images?" + params, {
          method: "POST", body: file,
        });
        if (result.added) uploaded += 1;
        else existing += 1;
      } catch (error) {
        if (!firstError) firstError = file.name + " — " + errorText(error);
      }
    }
    await refreshWorkspace(true);
    const summary = uploaded + " of " + files.length + " images uploaded.";
    const duplicates = existing ? existing + " already in this group." : "";
    showNotice("dataset-message", [summary, duplicates, firstError].filter(Boolean).join(" "), Boolean(firstError));
    $("dataset-images").value = "";
  } finally {
    datasetBusy = false;
    controls.forEach((control) => { control.disabled = false; });
  }
});

$("training-product").addEventListener("change", () => {
  showNotice("training-message");
  renderTraining();
});
$("start-training").addEventListener("submit", async (event) => {
  event.preventDefault();
  const product = selectedProduct("training-product");
  if (!product || trainingBusy) return;
  trainingBusy = true;
  $("training-submit").disabled = true;
  showNotice("training-message", "Submitting the training job…");
  try {
    const provider = $("training-provider").value;
    await api("/api/products/" + encodeURIComponent(product.id) + "/train", {
      method: "POST", body: { provider },
    });
    await refreshWorkspace(true);
    showNotice("training-message", "Training submitted. Follow progress under Training runs.");
  } catch (error) {
    showNotice("training-message", errorText(error), true);
  } finally {
    trainingBusy = false;
    renderTraining();
  }
});

async function activateVersion(productId, version, button) {
  button.disabled = true;
  showNotice("training-message");
  try {
    await api("/api/products/" + encodeURIComponent(productId) + "/activate", {
      method: "POST", body: { version_id: version.id },
    });
    await refreshWorkspace(true);
    showNotice("training-message", (version.name || version.id) + " is active for inspection.");
  } catch (error) {
    showNotice("training-message", errorText(error), true);
    button.disabled = false;
  }
}

window.addEventListener("hashchange", showPage);
showPage();
refreshWorkspace();
window.setInterval(() => {
  if (currentPage === "training" && !document.hidden) refreshWorkspace();
}, 4000);
