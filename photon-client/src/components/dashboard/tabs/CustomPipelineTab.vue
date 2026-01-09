<script setup lang="ts">
import { ref, watch, computed } from "vue";
import { useCameraSettingsStore } from "@/stores/settings/CameraSettingsStore";
import { useStateStore } from "@/stores/StateStore";

const store = useCameraSettingsStore();
const state = useStateStore();

const jsonText = ref("");
const parseError = ref<string | null>(null);
const validated = ref(false);
let parsedSettings = ref<unknown | null>(null);

// When the pipeline settings change, reflect them in the editor
watch(
  () => store.currentPipelineSettings,
  (v) => {
    try {
      jsonText.value = JSON.stringify(v, null, 2);
      parseError.value = null;
      parsedSettings.value = v;
      validated.value = true;
    } catch (e: any) {
      jsonText.value = "";
      parseError.value = "Failed to stringify current pipeline settings";
      parsedSettings.value = null;
      validated.value = false;
    }
  },
  { immediate: true }
);

function parseInputJson(input: string) {
  parseError.value = null;
  validated.value = false;
  parsedSettings.value = null;

  try {
    const parsed = JSON.parse(input);

    // Accept wrapper-array format ["TypeName", { ... }] or new object format { type, properties }
    let settings: any = parsed;
    if (Array.isArray(parsed) && parsed.length === 2 && typeof parsed[0] === "string") {
      settings = parsed[1];
    }

    // Validate basic shape: must be an object and for custom pipeline should include pipelineType or children
    if (settings == null || typeof settings !== "object") throw new Error("Parsed JSON must be an object or wrapper array");

    // If this is a top-level pipeline wrapper, descend into the 'pipeline' property
    if (settings.pipeline && typeof settings.pipeline === 'object') {
      settings = settings.pipeline;
    }

    // Minimal validation for sequential/parallel: must have children array
    if (!Array.isArray((settings as any).children) && (settings as any).pipelineType === undefined) {
      // Allow full-settings that don't include children (still valid) but mark as potentially incomplete
      // We'll accept any object though
    }

    parsedSettings.value = settings;
    validated.value = true;
    parseError.value = null;
  } catch (e: any) {
    parseError.value = e.message || String(e);
    parsedSettings.value = null;
    validated.value = false;
  }
}

function applyJson() {
  parseError.value = null;
  try {
    const parsed = JSON.parse(jsonText.value);
    // Support wrapper array
    const settings = Array.isArray(parsed) && parsed.length === 2 && typeof parsed[0] === "string" ? parsed[1] : parsed;
    parsedSettings.value = settings;
    validated.value = true;
    // Send all settings via changeCurrentPipelineSetting so the server will update fields
    store.changeCurrentPipelineSetting(settings, true);
    // show snackbar
    useStateStore().showSnackbarMessage({ color: "success", message: "Pipeline JSON applied" });
  } catch (e: any) {
    parseError.value = e.message || String(e);
    validated.value = false;
    useStateStore().showSnackbarMessage({ color: "error", message: "Invalid JSON: " + parseError.value });
  }
}

function onFileSelected(e: Event) {
  parseError.value = null;
  validated.value = false;
  parsedSettings.value = null;

  const el = e.target as HTMLInputElement;
  const file = el.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const text = String(reader.result);
      jsonText.value = text;
      parseInputJson(text);
    } catch (err: any) {
      parseError.value = err.message || String(err);
    }
  };
  reader.onerror = () => (parseError.value = "Failed to read file.");
  reader.readAsText(file);
}

function importToCurrentPipeline() {
  if (!validated.value || !parsedSettings.value) return;
  // Send the parsed settings to the backend as a full replacement for the current pipeline settings
  store.changeCurrentPipelineSetting(parsedSettings.value as any, true);
  useStateStore().showSnackbarMessage({ color: "success", message: "Imported JSON into current pipeline" });
}

// Simple helpers for visualizing the pipeline
import { PipelineType } from "@/types/PipelineTypes";
import PipelineVisualizer from "@/components/dashboard/PipelineVisualizer.vue";

function typeNameFromSettings(s: any): string {
  if (!s) return "(empty)";
  if (s.pipelineType !== undefined) {
    // pipelineType might be a PipelineType enum number
    const p = s.pipelineType as number;
    switch (p) {
      case PipelineType.Reflective:
        return "Reflective";
      case PipelineType.ColoredShape:
        return "Colored Shape";
      case PipelineType.AprilTag:
        return "AprilTag";
      case PipelineType.Aruco:
        return "ArUco";
      case PipelineType.ObjectDetection:
        return "Object Detection";
      case PipelineType.Sequential:
        return "Sequential";
      case PipelineType.Parallel:
        return "Parallel";
      default:
        return "Unknown(" + p + ")";
    }
  }
  // fallback heuristics
  if (s.children && Array.isArray(s.children)) {
    return s.children.length > 1 ? "Parallel?" : "Sequential?";
  }
  return "Custom";
}

// Recursive node builder using render function
function selectNode(s: any) {
  try {
    jsonText.value = JSON.stringify(s, null, 2);
    parsedSettings.value = s;
    validated.value = true;
    parseError.value = null;
  } catch (e: any) {
    parseError.value = "Failed to stringify node";
  }
}

function requestNodeFrame(path: number[]) {
  const payload = {
    changePipelineSetting: {
      requestNodeFrame: { path: path },
      cameraUniqueName: store.currentCameraSettings.uniqueName
    }
  };
  useStateStore().websocket?.send(payload, true);
}

function buildNode(s: any, path: number[] = []) {
  const title = typeNameFromSettings(s);
  const children = Array.isArray(s?.children) ? s.children : [];

  const childNodes = children.map((c: any, idx: number) => buildNode(c, path.concat([idx])));

  const iconMap: Record<string, string> = {
    "Reflective": "🔦",
    "Colored Shape": "◼️",
    "AprilTag": "#",
    "ArUco": "⬛",
    "Object Detection": "🎯",
    "Sequential": "🔁",
    "Parallel": "🔀",
    "Custom": "🧩"
  };

  const nodeIcon = iconMap[title] || "🧩";

  const baseProps: any = {
    class: "pv-node",
    onClick: (ev: Event) => {
      ev.stopPropagation();
      selectNode(s);
      requestNodeFrame(path);
    }
  };

  if (children.length === 0) {
    return h(
      "div",
      { class: "pv-node leaf", onClick: (ev: Event) => { ev.stopPropagation(); selectNode(s); requestNodeFrame(path); } },
      [h("div", { class: "pv-node-title" }, [h("span", { style: "margin-right:6px" }, nodeIcon), title])]
    );
  }

  // If settings suggest sequential, render inline
  const isSeq = s?.pipelineType === PipelineType.Sequential || (children.length > 0 && !s?.parallel);

  return h("div", baseProps, [
    h("div", { class: "pv-node-title" }, [h("span", { style: "margin-right:6px" }, nodeIcon), title]),
    h(
      "div",
      { class: isSeq ? "pv-node-children seq" : "pv-node-children par" },
      childNodes
    )
  ]);
}


</script>

<template>
  <div>
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px">
      <div style="font-weight:600">Custom JSON</div>
      <div style="opacity:0.8">Upload or edit the pipeline JSON and press Apply/Import to send it to the server.</div>
    </div>

    <div style="display:flex; gap:12px; margin-bottom:8px; align-items:flex-start">
      <div style="flex:1">
        <div style="display:flex; gap:8px; align-items:center; margin-bottom:6px">
          <input type="file" accept="application/json" @change="onFileSelected" />
          <button class="pv-btn" @click="parseInputJson(jsonText)">Validate</button>
          <button class="pv-btn pv-btn-primary" :disabled="!validated" @click="importToCurrentPipeline">Import</button>
          <div v-if="validated" style="color:var(--v-theme-success); margin-left:6px">Valid ✓</div>
          <div v-else-if="parseError" style="color:var(--v-theme-error); margin-left:6px">Error: {{ parseError }}</div>
        </div>
        <textarea v-model="jsonText" rows="12" style="width:100%; font-family: monospace; font-size: 12px"></textarea>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px">
          <div style="color:var(--v-theme-error)" v-if="parseError">Error: {{ parseError }}</div>
          <div style="flex:1;" />
          <button class="pv-btn pv-btn-primary" @click="applyJson">Apply</button>
        </div>
      </div>

    </div>

    <div style="margin-top:12px">
      <hr style="border:none; height:1px; background:rgba(0,0,0,0.06); margin:8px 0;" />
      <div style="margin-top:8px">
        <PipelineVisualizer />
      </div>
    </div>
  </div>
</template>

<style scoped>
.pv-node { border: 1px solid rgba(0,0,0,0.06); padding: 8px; border-radius: 6px; margin-bottom:8px; }
.pv-node-title { font-weight: 600; font-size: 12px; margin-bottom:6px }
.pv-node-children.seq { display:flex; gap:8px; flex-direction: row; align-items:flex-start }
.pv-node-children.par { display:flex; gap:8px; flex-direction: column }
.pv-node.leaf { background: rgba(0,0,0,0.02) }
.pv-node .pv-node { background: #fff; }
.pv-child { }
</style>
