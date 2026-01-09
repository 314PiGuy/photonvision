<script setup lang="ts">
import { h, computed, ref, onUnmounted, watch } from "vue";
import { useCameraSettingsStore } from "@/stores/settings/CameraSettingsStore";
import { useStateStore } from "@/stores/StateStore";
import { PipelineType } from "@/types/PipelineTypes";

const store = useCameraSettingsStore();
const state = useStateStore();

const props = defineProps<{ settings?: any; cameraName?: string }>();

const selectedPath = ref<number[] | null>(null);
let pollInterval: any = null;

function startPolling(path: number[]) {
  if (pollInterval) clearInterval(pollInterval);
  selectedPath.value = path;
  requestNodeFrame(path);
  pollInterval = setInterval(() => {
    requestNodeFrame(path);
  }, 100); // 10Hz
}

function stopPolling() {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = null;
  selectedPath.value = null;
  useStateStore().nodeFrame = undefined;
}

onUnmounted(() => {
  if (pollInterval) clearInterval(pollInterval);
});

// Pipeline type icons with distinctive visual representation
const pipelineIcons: Record<string, { icon: string; color: string }> = {
  "AprilTag": { icon: "mdi-tag", color: "#4CAF50" },
  "ArUco": { icon: "mdi-qrcode", color: "#9C27B0" },
  "Object Detection": { icon: "mdi-image-search", color: "#FF9800" },
  "Colored Shape": { icon: "mdi-shape", color: "#E91E63" },
  "Reflective": { icon: "mdi-flashlight", color: "#03A9F4" },
  "Sequential": { icon: "mdi-arrow-right-bold", color: "#607D8B" },
  "Parallel": { icon: "mdi-call-split", color: "#795548" },
  "Custom": { icon: "mdi-puzzle", color: "#9E9E9E" },
  "Unknown": { icon: "mdi-help-circle", color: "#757575" }
};

function getIconForType(title: string): { icon: string; color: string } {
  const t = String(title || "").toLowerCase();
  if (t.includes("april")) return pipelineIcons["AprilTag"];
  if (t.includes("aruco")) return pipelineIcons["ArUco"];
  if (t.includes("object") || t.includes("detection")) return pipelineIcons["Object Detection"];
  if (t.includes("colored") || t.includes("shape")) return pipelineIcons["Colored Shape"];
  if (t.includes("reflect")) return pipelineIcons["Reflective"];
  if (t.includes("sequent")) return pipelineIcons["Sequential"];
  if (t.includes("parallel")) return pipelineIcons["Parallel"];
  if (t.includes("custom")) return pipelineIcons["Custom"];
  return pipelineIcons["Unknown"];
}

function typeNameFromSettings(s: any): string {
  if (!s) return "(empty)";
  
  // Check for type property first (from JSON import format)
  if (s.type && typeof s.type === "string") {
    const typeStr = s.type.toLowerCase();
    if (typeStr.includes("object") || typeStr.includes("detection")) return "Object Detection";
    if (typeStr.includes("april")) return "AprilTag";
    if (typeStr.includes("aruco")) return "ArUco";
    if (typeStr.includes("reflect")) return "Reflective";
    if (typeStr.includes("colored") || typeStr.includes("shape")) return "Colored Shape";
    if (typeStr.includes("sequen")) return "Sequential";
    if (typeStr.includes("parallel")) return "Parallel";
    return s.type;
  }
  
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
    return s.children.length > 1 ? "Parallel" : "Sequential";
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
function normalizeNode(input: any): { title: string; payload: any; children: any[]; properties?: any } {
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

  // Envelope form { type: "...", properties: {...}, children: [...] }
  if (input.type && typeof input.type === "string") {
    const childrenRaw = input.children || input.properties?.children || [];
    const children = Array.isArray(childrenRaw) ? childrenRaw.map((c: any) => normalizeNode(c)) : [];
    return { 
      title: typeNameFromSettings(input), 
      payload: input.properties || input, 
      children,
      properties: input.properties
    };
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

// Check if the root pipeline is selected (no node selected)
const isRootSelected = computed(() => selectedPath.value === null);
</script>

<template>
  <div class="pipeline-visualizer">
    <div class="pv-header">
      <v-icon size="small" class="mr-2">mdi-sitemap</v-icon>
      <span class="pv-title">Pipeline Structure</span>
    </div>
    <div class="pv-description">
      Click a node to view its output stream and targets. Click the camera to view final output.
    </div>

    <div v-if="rootNode" class="pv-tree-container">
      <!-- Camera root node -->
      <div 
        class="pv-node pv-camera-node" 
        :class="{ selected: isRootSelected }"
        @click="stopPolling"
      >
        <v-icon size="20" color="#2196F3" class="pv-node-icon">mdi-camera</v-icon>
        <span class="pv-node-label">{{ cameraLabel }}</span>
        <v-chip v-if="isRootSelected" size="x-small" color="primary" class="ml-2">Active</v-chip>
      </div>

      <!-- Pipeline tree -->
      <div class="pv-tree">
        <PipelineTreeNode
          :node="rootNode"
          :path="[]"
          :selectedPath="selectedPath"
          :getIconForType="getIconForType"
          @select="startPolling"
        />
      </div>
    </div>
    <div v-else class="pv-empty">
      <v-icon size="32" color="grey">mdi-alert-circle-outline</v-icon>
      <span>No pipeline to visualize</span>
    </div>
  </div>
</template>

<script lang="ts">
// Recursive tree node component
import { defineComponent, type PropType } from "vue";

const PipelineTreeNode = defineComponent({
  name: "PipelineTreeNode",
  props: {
    node: { type: Object as PropType<any>, required: true },
    path: { type: Array as PropType<number[]>, required: true },
    selectedPath: { type: Array as PropType<number[] | null>, default: null },
    getIconForType: { type: Function as PropType<(title: string) => { icon: string; color: string }>, required: true },
    depth: { type: Number, default: 0 }
  },
  emits: ["select"],
  setup(props, { emit }) {
    const isSelected = computed(() => {
      if (!props.selectedPath) return false;
      return JSON.stringify(props.selectedPath) === JSON.stringify(props.path);
    });

    const iconInfo = computed(() => props.getIconForType(props.node.title));
    
    const hasChildren = computed(() => props.node.children && props.node.children.length > 0);
    
    const isParallel = computed(() => {
      const title = props.node.title?.toLowerCase() || "";
      return title.includes("parallel");
    });

    const handleClick = (ev: Event) => {
      ev.stopPropagation();
      emit("select", props.path);
    };

    return { isSelected, iconInfo, hasChildren, isParallel, handleClick };
  },
  template: `
    <div class="pv-tree-node" :class="{ 'has-children': hasChildren }">
      <div 
        class="pv-node" 
        :class="{ selected: isSelected, leaf: !hasChildren }"
        @click="handleClick"
      >
        <v-icon size="18" :color="iconInfo.color" class="pv-node-icon">{{ iconInfo.icon }}</v-icon>
        <span class="pv-node-label">{{ node.title }}</span>
        <v-chip v-if="isSelected" size="x-small" color="primary" class="ml-2">Viewing</v-chip>
      </div>
      <div v-if="hasChildren" class="pv-children" :class="{ parallel: isParallel, sequential: !isParallel }">
        <div class="pv-connector" :class="{ parallel: isParallel }"></div>
        <PipelineTreeNode
          v-for="(child, idx) in node.children"
          :key="idx"
          :node="child"
          :path="[...path, idx]"
          :selectedPath="selectedPath"
          :getIconForType="getIconForType"
          :depth="depth + 1"
          @select="$emit('select', $event)"
        />
      </div>
    </div>
  `
});

export default {
  components: { PipelineTreeNode }
};
</script>

<style scoped>
.pipeline-visualizer {
  background: var(--v-theme-surface);
  border-radius: 8px;
  padding: 12px;
}

.pv-header {
  display: flex;
  align-items: center;
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 4px;
}

.pv-title {
  color: var(--v-theme-on-surface);
}

.pv-description {
  font-size: 12px;
  color: rgba(var(--v-theme-on-surface), 0.6);
  margin-bottom: 12px;
}

.pv-tree-container {
  background: rgba(0, 0, 0, 0.02);
  border-radius: 6px;
  padding: 12px;
}

.pv-tree {
  margin-left: 16px;
  margin-top: 8px;
  border-left: 2px solid rgba(0, 0, 0, 0.08);
  padding-left: 12px;
}

.pv-node {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 6px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  background: var(--v-theme-surface);
  cursor: pointer;
  transition: all 0.15s ease;
  margin-bottom: 6px;
}

.pv-node:hover {
  border-color: rgba(var(--v-theme-primary), 0.4);
  background: rgba(var(--v-theme-primary), 0.05);
}

.pv-node.selected {
  border-color: rgb(var(--v-theme-primary));
  background: rgba(var(--v-theme-primary), 0.1);
  box-shadow: 0 2px 8px rgba(var(--v-theme-primary), 0.15);
}

.pv-node.leaf {
  background: rgba(0, 0, 0, 0.02);
}

.pv-camera-node {
  background: rgba(33, 150, 243, 0.08);
  border-color: rgba(33, 150, 243, 0.2);
}

.pv-camera-node:hover {
  border-color: rgba(33, 150, 243, 0.5);
}

.pv-camera-node.selected {
  border-color: #2196F3;
  background: rgba(33, 150, 243, 0.15);
}

.pv-node-icon {
  flex-shrink: 0;
}

.pv-node-label {
  font-weight: 500;
  font-size: 13px;
  color: var(--v-theme-on-surface);
}

.pv-tree-node {
  position: relative;
}

.pv-tree-node.has-children > .pv-node {
  font-weight: 600;
}

.pv-children {
  margin-left: 20px;
  margin-top: 4px;
  padding-left: 12px;
  border-left: 2px dashed rgba(0, 0, 0, 0.1);
  position: relative;
}

.pv-children.parallel {
  border-left-style: solid;
  border-left-color: rgba(121, 85, 72, 0.3);
}

.pv-children.sequential {
  border-left-color: rgba(96, 125, 139, 0.3);
}

.pv-connector {
  position: absolute;
  left: -14px;
  top: 16px;
  width: 12px;
  height: 2px;
  background: rgba(0, 0, 0, 0.1);
}

.pv-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 24px;
  opacity: 0.6;
  font-size: 13px;
}
</style>
