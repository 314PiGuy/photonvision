<script setup lang="ts">
import { defineComponent, h } from "vue";
import { useCameraSettingsStore } from "@/stores/settings/CameraSettingsStore";
import { useStateStore } from "@/stores/StateStore";
import { PipelineType } from "@/types/PipelineTypes";

const store = useCameraSettingsStore();
const state = useStateStore();

const props = defineProps<{ settings?: any; cameraName?: string }>();

import { computed } from "vue";

function typeNameFromSettings(s: any): string {
  if (!s) return "(empty)";
  if (s.pipelineType !== undefined) {
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
  if (s.children && Array.isArray(s.children)) {
    return s.children.length > 1 ? "Parallel?" : "Sequential?";
  }
  return "Custom";
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

// Normalize various pipeline input shapes into a simple node structure
function normalizeNode(input: any) {
  if (!input) return { title: "(empty)", payload: input, children: [] };

  // Top-level JSON may be { pipeline: { ... } }
  if (input.pipeline) input = input.pipeline;

  // Wrapper-array form ["TypeName", payload]
  if (Array.isArray(input) && input.length === 2 && typeof input[0] === "string") {
    const typeName = input[0];
    const payload = input[1];
    const childrenRaw = payload?.children || payload?.pipeline?.children || [];
    const children = Array.isArray(childrenRaw) ? childrenRaw.map((c: any) => normalizeNode(c)) : [];
    return { title: typeName, payload, children };
  }

  // Envelope form { type: "...", payload: {...} }
  if (input.type && input.payload !== undefined) {
    return normalizeNode([String(input.type), input.payload]);
  }

  // Object form - try to derive children and type name
  const childrenRaw = input?.children || [];
  const children = Array.isArray(childrenRaw) ? childrenRaw.map((c: any) => normalizeNode(c)) : [];
  const t = typeNameFromSettings(input);
  return { title: t, payload: input, children };
}

const rootNode = computed(() => {
  const s = props.settings || store.currentPipelineSettings;
  if (!s) return null;
  return normalizeNode(s);
});

const cameraLabel = computed(() => props.settings?.sourceCamera || props.cameraName || store.currentCameraSettings?.nickname || store.currentCameraSettings?.uniqueName || "Camera");

// Render normalized node into VNode recursively
function renderNode(node: any, path: number[] = []) {
  if (!node) return h("div", "(empty)");

  function iconForTitle(title: string) {
    const t = String(title || "").toLowerCase();
    if (t.includes("april")) return "🏷️";
    if (t.includes("aruco")) return "⬛";
    if (t.includes("object") || t.includes("detection")) return "🎯";
    if (t.includes("colored")) return "◼️";
    if (t.includes("reflect")) return "🔦";
    if (t.includes("sequent")) return "🔁";
    if (t.includes("parallel")) return "🔀";
    return "🧩";
  }

  const title = node.title || "(unknown)";
  const icon = iconForTitle(title);

  const children = node.children || [];

  const onClick = (ev: Event) => {
    ev.stopPropagation();
    requestNodeFrame(path);
  };

  const childVNodes = children.map((c: any, idx: number) => h(
    "li",
    { class: "pv-li" },
    [h("component", { is: { render: () => renderNode(c, path.concat([idx])) } })]
  ));

  return h("div", { class: "pv-node", onClick }, [
    h("span", { class: "pv-icon" }, icon),
    h("span", { class: "pv-title" }, title),
    children.length ? h("ul", { class: "pv-node-children" }, childVNodes) : null
  ]);
}

</script>

<template>
  <div>
    <div style="font-weight:600; margin-bottom:6px">Visualizer</div>
    <div style="font-size:12px; color:var(--v-theme-on-surface); margin-bottom:8px">
      A simple representation of the current pipeline structure. Click a node to request its debug frame.
    </div>

    <div v-if="rootNode">
      <ul class="pv-tree">
        <li class="pv-li root">
          <div class="pv-node-inline root">
            <span class="pv-icon">📷</span>
            <span class="pv-title">{{ cameraLabel }}</span>
          </div>
          <ul class="pv-tree">
            <li>
              <component :is="{ render: () => renderNode(rootNode.value) }" />
            </li>
          </ul>
        </li>
      </ul>
    </div>
    <div v-else style="opacity:0.6; margin-top:8px">No pipeline to visualize</div>
  </div>
</template>

<style scoped>
.pv-tree { list-style:none; margin:0; padding-left: 12px; }
.pv-tree .pv-tree { padding-left: 18px; }
.pv-li { position: relative; margin: 0; padding-left: 18px; }
.pv-li::before { content: ''; position: absolute; left: 6px; top: 0; bottom: 0; width: 1px; background: rgba(0,0,0,0.06); }
.pv-li > ul > li::before { display: none; }
.pv-node-inline { display:flex; align-items:center; gap:8px; padding:6px 8px; border-radius:6px; }
.pv-node { padding:6px 8px; border-radius:6px; border: 1px solid rgba(0,0,0,0.06); background: var(--v-theme-surface); display:inline-block }
.pv-node .pv-title { font-weight:600; font-size:13px }
.pv-node .pv-icon { width:20px; display:inline-block }
.pv-node.leaf { background: rgba(0,0,0,0.02) }
.pv-node-children { margin-top:8px }
.pv-node-children.seq { display:flex; gap:12px; flex-direction: row; align-items:flex-start }
.pv-node-children.par { display:flex; gap:12px; flex-direction: column }
</style>
