<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from "vue";
import { useStereoStore, type StereoConfiguration, type StereoTarget } from "@/stores/StereoStore";
import { useStateStore } from "@/stores/StateStore";
import { useTheme } from "vuetify";

const stereoStore = useStereoStore();
const stateStore = useStateStore();
const theme = useTheme();

// Refresh timer
let refreshTimer: ReturnType<typeof setInterval> | null = null;

// Configuration form
const showConfigDialog = ref(false);
const configFileInput = ref<HTMLInputElement | null>(null);
const configJson = ref("");
const configError = ref("");

// Form fields for manual configuration
const formLeftCamera = ref("");
const formRightCamera = ref("");
const formLeftX = ref(-0.15);
const formRightX = ref(0.15);
const formLeftY = ref(0);
const formRightY = ref(0);
const formLeftZ = ref(0);
const formRightZ = ref(0);
const formLeftRotX = ref(0);
const formLeftRotY = ref(0);
const formLeftRotZ = ref(0);
const formRightRotX = ref(0);
const formRightRotY = ref(0);
const formRightRotZ = ref(0);
const formLeftHFov = ref(70);
const formLeftVFov = ref(50);
const formRightHFov = ref(70);
const formRightVFov = ref(50);

const configMode = ref<"form" | "json">("form");

// Computed properties
const hasActivePair = computed(() => stereoStore.hasActivePair);
const currentResult = computed(() => stereoStore.currentResult);
const currentTargets = computed(() => stereoStore.currentTargets);
const stereoImage = computed(() => stereoStore.stereoImage);
const availableCameras = computed(() => stereoStore.availableCameras);
const activePairs = computed(() => stereoStore.activePairs);
const isLoading = computed(() => stereoStore.isLoading);
const error = computed(() => stereoStore.error);

// Table headers for targets
const targetHeaders = [
  { title: "#", key: "matchId", width: "60px" },
  { title: "Class", key: "className" },
  { title: "Confidence", key: "confidence" },
  { title: "Depth (m)", key: "depth" },
  { title: "X Offset (m)", key: "perpendicularDistance" },
  { title: "Y Offset (m)", key: "verticalOffset" },
  { title: "Match Quality", key: "matchQuality" }
];

// Format number for display
const formatNumber = (value: number, decimals = 2): string => {
  if (value === undefined || value === null || !isFinite(value)) return "N/A";
  return value.toFixed(decimals);
};

// Initialize
onMounted(async () => {
  await stereoStore.fetchAvailableCameras();
  await stereoStore.fetchActivePairs();
  startRefresh();
});

onUnmounted(() => {
  stopRefresh();
});

// Watch for auto-refresh changes
watch(
  () => stereoStore.autoRefresh,
  (enabled) => {
    if (enabled) {
      startRefresh();
    } else {
      stopRefresh();
    }
  }
);

function startRefresh() {
  if (refreshTimer) return;
  refreshTimer = setInterval(async () => {
    if (stereoStore.autoRefresh && hasActivePair.value) {
      await stereoStore.fetchResult();
      await stereoStore.fetchStereoImage();
    }
  }, stereoStore.refreshInterval);
}

function stopRefresh() {
  if (refreshTimer) {
    clearInterval(refreshTimer);
    refreshTimer = null;
  }
}

// Configuration handling
function openConfigDialog() {
  configError.value = "";
  configJson.value = "";
  showConfigDialog.value = true;
  
  // Set default cameras if available
  if (availableCameras.value.length >= 2) {
    formLeftCamera.value = availableCameras.value[0];
    formRightCamera.value = availableCameras.value[1];
  }
}

function triggerFileUpload() {
  configFileInput.value?.click();
}

async function handleFileUpload(event: Event) {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];
  if (!file) return;

  const success = await stereoStore.uploadConfigurationFile(file);
  if (success) {
    showConfigDialog.value = false;
    stateStore.showSnackbarMessage({
      message: "Stereo pair created successfully",
      color: "success"
    });
  } else {
    configError.value = stereoStore.error || "Failed to upload configuration";
  }
}

async function submitFormConfig() {
  if (!formLeftCamera.value || !formRightCamera.value) {
    configError.value = "Please select both cameras";
    return;
  }
  if (formLeftCamera.value === formRightCamera.value) {
    configError.value = "Left and right cameras must be different";
    return;
  }

  const config: StereoConfiguration = {
    leftCameraName: formLeftCamera.value,
    rightCameraName: formRightCamera.value,
    leftPosition: {
      x: formLeftX.value,
      y: formLeftY.value,
      z: formLeftZ.value,
      rotX: formLeftRotX.value,
      rotY: formLeftRotY.value,
      rotZ: formLeftRotZ.value
    },
    rightPosition: {
      x: formRightX.value,
      y: formRightY.value,
      z: formRightZ.value,
      rotX: formRightRotX.value,
      rotY: formRightRotY.value,
      rotZ: formRightRotZ.value
    },
    leftFOV: {
      horizontalFOV: formLeftHFov.value,
      verticalFOV: formLeftVFov.value
    },
    rightFOV: {
      horizontalFOV: formRightHFov.value,
      verticalFOV: formRightVFov.value
    }
  };

  const success = await stereoStore.uploadConfiguration(config);
  if (success) {
    showConfigDialog.value = false;
    stateStore.showSnackbarMessage({
      message: "Stereo pair created successfully",
      color: "success"
    });
  } else {
    configError.value = stereoStore.error || "Failed to create stereo pair";
  }
}

async function submitJsonConfig() {
  if (!configJson.value.trim()) {
    configError.value = "Please enter JSON configuration";
    return;
  }

  try {
    const config = JSON.parse(configJson.value);
    const success = await stereoStore.uploadConfiguration(config);
    if (success) {
      showConfigDialog.value = false;
      stateStore.showSnackbarMessage({
        message: "Stereo pair created successfully",
        color: "success"
      });
    } else {
      configError.value = stereoStore.error || "Failed to create stereo pair";
    }
  } catch (e) {
    configError.value = "Invalid JSON format";
  }
}

async function removePair(pairId: string) {
  const success = await stereoStore.removePair(pairId);
  if (success) {
    stateStore.showSnackbarMessage({
      message: "Stereo pair removed",
      color: "success"
    });
  }
}

async function downloadTemplate() {
  const template = await stereoStore.fetchConfigTemplate();
  if (template) {
    const json = JSON.stringify(template, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "stereo_config_template.json";
    a.click();
    URL.revokeObjectURL(url);
  }
}

// Get row color based on match quality
function getRowClass(target: StereoTarget): string {
  if (target.matchQuality < 0.1) return "good-match";
  if (target.matchQuality < 0.3) return "medium-match";
  return "poor-match";
}
</script>

<template>
  <v-container class="pa-3" fluid>
    <!-- Header -->
    <v-row class="mb-3">
      <v-col cols="12">
        <v-card color="primary" class="pa-3">
          <div class="d-flex align-center justify-space-between">
            <div class="d-flex align-center">
              <v-icon size="large" class="mr-3">mdi-camera-burst</v-icon>
              <div>
                <h2 class="text-h5">Stereo Vision</h2>
                <span class="text-subtitle-2">3D object detection with stereo camera pairs</span>
              </div>
            </div>
            <div class="d-flex align-center gap-2">
              <v-btn color="white" variant="outlined" @click="downloadTemplate">
                <v-icon left>mdi-download</v-icon>
                Template
              </v-btn>
              <v-btn color="white" @click="openConfigDialog">
                <v-icon left>mdi-plus</v-icon>
                Add Pair
              </v-btn>
            </div>
          </div>
        </v-card>
      </v-col>
    </v-row>

    <!-- No pairs message -->
    <v-row v-if="!hasActivePair" class="mb-3">
      <v-col cols="12">
        <v-card class="pa-6 text-center">
          <v-icon size="64" color="grey">mdi-camera-off</v-icon>
          <h3 class="text-h6 mt-4">No Stereo Pairs Configured</h3>
          <p class="text-body-2 mt-2 mb-4">
            Upload a JSON configuration file to create a stereo camera pair for 3D object detection.
          </p>
          <v-btn color="primary" @click="openConfigDialog">
            <v-icon left>mdi-plus</v-icon>
            Add Stereo Pair
          </v-btn>
        </v-card>
      </v-col>
    </v-row>

    <!-- Main content when pairs exist -->
    <template v-if="hasActivePair">
      <!-- Status bar -->
      <v-row class="mb-3">
        <v-col cols="12">
          <v-card class="pa-3">
            <div class="d-flex align-center justify-space-between flex-wrap gap-3">
              <!-- Active pairs selector -->
              <div class="d-flex align-center gap-3">
                <v-chip-group v-model="stereoStore.currentPairId" mandatory>
                  <v-chip
                    v-for="pairId in activePairs"
                    :key="pairId"
                    :value="pairId"
                    filter
                    @click="stereoStore.selectPair(pairId)"
                  >
                    {{ pairId }}
                    <v-btn
                      icon
                      size="x-small"
                      variant="text"
                      class="ml-1"
                      @click.stop="removePair(pairId)"
                    >
                      <v-icon size="small">mdi-close</v-icon>
                    </v-btn>
                  </v-chip>
                </v-chip-group>
              </div>

              <!-- Stats -->
              <div class="d-flex align-center gap-4">
                <v-chip v-if="currentResult?.isValid" color="success" size="small">
                  <v-icon start size="small">mdi-check-circle</v-icon>
                  Active
                </v-chip>
                <v-chip v-else color="warning" size="small">
                  <v-icon start size="small">mdi-alert</v-icon>
                  Waiting
                </v-chip>
                
                <span class="text-body-2">
                  <v-icon size="small">mdi-speedometer</v-icon>
                  {{ formatNumber(currentResult?.fps || 0, 1) }} FPS
                </span>
                
                <span class="text-body-2">
                  <v-icon size="small">mdi-timer</v-icon>
                  {{ formatNumber(currentResult?.processingTimeMs || 0, 1) }} ms
                </span>
                
                <span class="text-body-2">
                  <v-icon size="small">mdi-target</v-icon>
                  {{ currentResult?.matchCount || 0 }} matches
                </span>

                <v-switch
                  v-model="stereoStore.autoRefresh"
                  hide-details
                  density="compact"
                  label="Auto-refresh"
                  class="ml-2"
                />
              </div>
            </div>
          </v-card>
        </v-col>
      </v-row>

      <!-- Stereo view and targets -->
      <v-row>
        <!-- Stereo image -->
        <v-col cols="12" lg="8">
          <v-card class="pa-3">
            <h3 class="text-h6 mb-3">Stereo View</h3>
            <div class="stereo-view-container">
              <img
                v-if="stereoImage"
                :src="'data:image/jpeg;base64,' + stereoImage"
                alt="Stereo view"
                class="stereo-image"
              />
              <div v-else class="stereo-placeholder">
                <v-icon size="64" color="grey">mdi-image-off</v-icon>
                <p class="mt-2">Waiting for stereo frames...</p>
              </div>
            </div>
          </v-card>
        </v-col>

        <!-- Detection info -->
        <v-col cols="12" lg="4">
          <v-card class="pa-3 mb-3">
            <h3 class="text-h6 mb-3">Detection Summary</h3>
            <v-row dense>
              <v-col cols="6">
                <v-card variant="outlined" class="pa-3 text-center">
                  <div class="text-h4 text-primary">{{ currentResult?.leftDetections || 0 }}</div>
                  <div class="text-caption">Left Camera</div>
                </v-card>
              </v-col>
              <v-col cols="6">
                <v-card variant="outlined" class="pa-3 text-center">
                  <div class="text-h4 text-primary">{{ currentResult?.rightDetections || 0 }}</div>
                  <div class="text-caption">Right Camera</div>
                </v-card>
              </v-col>
              <v-col cols="12">
                <v-card variant="outlined" class="pa-3 text-center" color="success">
                  <div class="text-h4">{{ currentResult?.matchCount || 0 }}</div>
                  <div class="text-caption">Matched Pairs</div>
                </v-card>
              </v-col>
            </v-row>
          </v-card>

          <v-card class="pa-3">
            <h3 class="text-h6 mb-3">Configuration</h3>
            <v-list density="compact">
              <v-list-item v-if="stereoStore.currentConfiguration">
                <template #prepend>
                  <v-icon>mdi-camera</v-icon>
                </template>
                <v-list-item-title>Left Camera</v-list-item-title>
                <v-list-item-subtitle>{{ stereoStore.currentConfiguration.leftCameraName }}</v-list-item-subtitle>
              </v-list-item>
              <v-list-item v-if="stereoStore.currentConfiguration">
                <template #prepend>
                  <v-icon>mdi-camera</v-icon>
                </template>
                <v-list-item-title>Right Camera</v-list-item-title>
                <v-list-item-subtitle>{{ stereoStore.currentConfiguration.rightCameraName }}</v-list-item-subtitle>
              </v-list-item>
            </v-list>
          </v-card>
        </v-col>
      </v-row>

      <!-- Targets table -->
      <v-row class="mt-3">
        <v-col cols="12">
          <v-card class="pa-3">
            <h3 class="text-h6 mb-3">Detected Targets</h3>
            <v-data-table
              :headers="targetHeaders"
              :items="currentTargets"
              :items-per-page="10"
              class="elevation-1"
              density="compact"
            >
              <template #item.confidence="{ item }">
                {{ formatNumber(item.confidence * 100, 1) }}%
              </template>
              <template #item.depth="{ item }">
                {{ formatNumber(item.depth, 2) }}
              </template>
              <template #item.perpendicularDistance="{ item }">
                {{ formatNumber(item.perpendicularDistance, 2) }}
              </template>
              <template #item.verticalOffset="{ item }">
                {{ formatNumber(item.verticalOffset, 2) }}
              </template>
              <template #item.matchQuality="{ item }">
                <v-chip
                  :color="item.matchQuality < 0.1 ? 'success' : item.matchQuality < 0.3 ? 'warning' : 'error'"
                  size="small"
                >
                  {{ formatNumber(item.matchQuality, 3) }}
                </v-chip>
              </template>
              <template #no-data>
                <div class="text-center pa-4">
                  <v-icon size="large" color="grey">mdi-target-variant</v-icon>
                  <p class="mt-2">No matched targets detected</p>
                </div>
              </template>
            </v-data-table>
          </v-card>
        </v-col>
      </v-row>
    </template>

    <!-- Configuration dialog -->
    <v-dialog v-model="showConfigDialog" max-width="800">
      <v-card>
        <v-card-title class="text-h5">
          <v-icon left>mdi-camera-burst</v-icon>
          Add Stereo Pair
        </v-card-title>
        
        <v-card-text>
          <v-alert v-if="configError" type="error" class="mb-4" closable @click:close="configError = ''">
            {{ configError }}
          </v-alert>

          <v-tabs v-model="configMode" class="mb-4">
            <v-tab value="form">Form</v-tab>
            <v-tab value="json">JSON</v-tab>
          </v-tabs>

          <v-window v-model="configMode">
            <!-- Form mode -->
            <v-window-item value="form">
              <v-row>
                <v-col cols="6">
                  <v-select
                    v-model="formLeftCamera"
                    :items="availableCameras"
                    label="Left Camera"
                    prepend-icon="mdi-camera"
                  />
                </v-col>
                <v-col cols="6">
                  <v-select
                    v-model="formRightCamera"
                    :items="availableCameras"
                    label="Right Camera"
                    prepend-icon="mdi-camera"
                  />
                </v-col>
              </v-row>

              <v-divider class="my-4" />
              <h4 class="text-subtitle-1 mb-3">Camera Positions (meters from center)</h4>
              
              <v-row>
                <v-col cols="6">
                  <h5 class="text-subtitle-2 mb-2">Left Camera</h5>
                  <v-row dense>
                    <v-col cols="4">
                      <v-text-field v-model.number="formLeftX" label="X" type="number" step="0.01" density="compact" />
                    </v-col>
                    <v-col cols="4">
                      <v-text-field v-model.number="formLeftY" label="Y" type="number" step="0.01" density="compact" />
                    </v-col>
                    <v-col cols="4">
                      <v-text-field v-model.number="formLeftZ" label="Z" type="number" step="0.01" density="compact" />
                    </v-col>
                  </v-row>
                </v-col>
                <v-col cols="6">
                  <h5 class="text-subtitle-2 mb-2">Right Camera</h5>
                  <v-row dense>
                    <v-col cols="4">
                      <v-text-field v-model.number="formRightX" label="X" type="number" step="0.01" density="compact" />
                    </v-col>
                    <v-col cols="4">
                      <v-text-field v-model.number="formRightY" label="Y" type="number" step="0.01" density="compact" />
                    </v-col>
                    <v-col cols="4">
                      <v-text-field v-model.number="formRightZ" label="Z" type="number" step="0.01" density="compact" />
                    </v-col>
                  </v-row>
                </v-col>
              </v-row>

              <v-divider class="my-4" />
              <h4 class="text-subtitle-1 mb-3">Field of View (degrees)</h4>
              
              <v-row>
                <v-col cols="6">
                  <h5 class="text-subtitle-2 mb-2">Left Camera</h5>
                  <v-row dense>
                    <v-col cols="6">
                      <v-text-field v-model.number="formLeftHFov" label="Horizontal FOV" type="number" density="compact" />
                    </v-col>
                    <v-col cols="6">
                      <v-text-field v-model.number="formLeftVFov" label="Vertical FOV" type="number" density="compact" />
                    </v-col>
                  </v-row>
                </v-col>
                <v-col cols="6">
                  <h5 class="text-subtitle-2 mb-2">Right Camera</h5>
                  <v-row dense>
                    <v-col cols="6">
                      <v-text-field v-model.number="formRightHFov" label="Horizontal FOV" type="number" density="compact" />
                    </v-col>
                    <v-col cols="6">
                      <v-text-field v-model.number="formRightVFov" label="Vertical FOV" type="number" density="compact" />
                    </v-col>
                  </v-row>
                </v-col>
              </v-row>
            </v-window-item>

            <!-- JSON mode -->
            <v-window-item value="json">
              <v-textarea
                v-model="configJson"
                label="JSON Configuration"
                placeholder='{"leftCameraName": "camera1", "rightCameraName": "camera2", ...}'
                rows="12"
                variant="outlined"
                class="mono-text"
              />
              <div class="d-flex gap-2">
                <v-btn variant="text" size="small" @click="triggerFileUpload">
                  <v-icon left>mdi-upload</v-icon>
                  Upload File
                </v-btn>
                <v-btn variant="text" size="small" @click="downloadTemplate">
                  <v-icon left>mdi-download</v-icon>
                  Download Template
                </v-btn>
              </div>
              <input
                ref="configFileInput"
                type="file"
                accept=".json"
                hidden
                @change="handleFileUpload"
              />
            </v-window-item>
          </v-window>
        </v-card-text>

        <v-card-actions>
          <v-spacer />
          <v-btn text @click="showConfigDialog = false">Cancel</v-btn>
          <v-btn
            color="primary"
            :loading="isLoading"
            @click="configMode === 'form' ? submitFormConfig() : submitJsonConfig()"
          >
            Create Pair
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<style scoped>
.stereo-view-container {
  width: 100%;
  background: #1a1a1a;
  border-radius: 8px;
  overflow: hidden;
  min-height: 400px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.stereo-image {
  width: 100%;
  height: auto;
  display: block;
}

.stereo-placeholder {
  text-align: center;
  color: #888;
  padding: 40px;
}

.mono-text :deep(textarea) {
  font-family: monospace;
}

.good-match {
  background-color: rgba(76, 175, 80, 0.1);
}

.medium-match {
  background-color: rgba(255, 193, 7, 0.1);
}

.poor-match {
  background-color: rgba(244, 67, 54, 0.1);
}

.gap-2 {
  gap: 8px;
}

.gap-3 {
  gap: 12px;
}

.gap-4 {
  gap: 16px;
}
</style>
