import { defineStore } from "pinia";
import axios from "axios";

export interface StereoTarget {
  matchId: number;
  className: string;
  classIdx: number;
  confidence: number;
  depth: number;
  perpendicularDistance: number;
  verticalOffset: number;
  matchQuality: number;
}

export interface StereoResult {
  isValid: boolean;
  fps: number;
  processingTimeMs: number;
  leftDetections: number;
  rightDetections: number;
  matchCount: number;
  targets: StereoTarget[];
}

export interface CameraPosition {
  x: number;
  y: number;
  z: number;
  rotX: number;
  rotY: number;
  rotZ: number;
}

export interface CameraFOV {
  horizontalFOV: number;
  verticalFOV: number;
}

export interface StereoConfiguration {
  leftCameraName: string;
  rightCameraName: string;
  leftPosition: CameraPosition;
  rightPosition: CameraPosition;
  leftFOV: CameraFOV;
  rightFOV: CameraFOV;
}

interface StereoStoreState {
  availableCameras: string[];
  activePairs: string[];
  currentPairId: string | null;
  currentResult: StereoResult | null;
  currentConfiguration: StereoConfiguration | null;
  stereoImage: string | null;
  isLoading: boolean;
  error: string | null;
  autoRefresh: boolean;
  refreshInterval: number;
}

export const useStereoStore = defineStore("stereo", {
  state: (): StereoStoreState => ({
    availableCameras: [],
    activePairs: [],
    currentPairId: null,
    currentResult: null,
    currentConfiguration: null,
    stereoImage: null,
    isLoading: false,
    error: null,
    autoRefresh: true,
    refreshInterval: 100 // ms
  }),

  getters: {
    hasActivePair(): boolean {
      return this.activePairs.length > 0;
    },
    currentTargets(): StereoTarget[] {
      return this.currentResult?.targets ?? [];
    }
  },

  actions: {
    async fetchAvailableCameras() {
      try {
        const response = await axios.get("/stereo/cameras");
        this.availableCameras = response.data;
      } catch (error) {
        console.error("Failed to fetch available cameras:", error);
        this.error = "Failed to fetch available cameras";
      }
    },

    async fetchActivePairs() {
      try {
        const response = await axios.get("/stereo/pairs");
        this.activePairs = response.data;
        if (this.activePairs.length > 0 && !this.currentPairId) {
          this.currentPairId = this.activePairs[0];
        }
      } catch (error) {
        console.error("Failed to fetch active pairs:", error);
        this.error = "Failed to fetch active pairs";
      }
    },

    async uploadConfiguration(config: StereoConfiguration) {
      this.isLoading = true;
      this.error = null;
      try {
        const response = await axios.post("/stereo/config", config);
        this.currentPairId = response.data.pairId;
        await this.fetchActivePairs();
        return true;
      } catch (error) {
        console.error("Failed to upload configuration:", error);
        this.error = "Failed to upload configuration";
        return false;
      } finally {
        this.isLoading = false;
      }
    },

    async uploadConfigurationFile(file: File) {
      this.isLoading = true;
      this.error = null;
      try {
        const formData = new FormData();
        formData.append("config", file);
        const response = await axios.post("/stereo/config", formData, {
          headers: { "Content-Type": "multipart/form-data" }
        });
        this.currentPairId = response.data.pairId;
        await this.fetchActivePairs();
        return true;
      } catch (error) {
        console.error("Failed to upload configuration file:", error);
        this.error = "Failed to upload configuration file";
        return false;
      } finally {
        this.isLoading = false;
      }
    },

    async removePair(pairId: string) {
      try {
        await axios.delete(`/stereo/pair?pairId=${encodeURIComponent(pairId)}`);
        await this.fetchActivePairs();
        if (this.currentPairId === pairId) {
          this.currentPairId = this.activePairs[0] ?? null;
        }
        return true;
      } catch (error) {
        console.error("Failed to remove pair:", error);
        this.error = "Failed to remove stereo pair";
        return false;
      }
    },

    async fetchResult() {
      try {
        const params = this.currentPairId ? `?pairId=${encodeURIComponent(this.currentPairId)}` : "";
        const response = await axios.get(`/stereo/result${params}`);
        this.currentResult = response.data;
      } catch (error) {
        console.error("Failed to fetch stereo result:", error);
      }
    },

    async fetchConfiguration(pairId: string) {
      try {
        const response = await axios.get(`/stereo/configuration?pairId=${encodeURIComponent(pairId)}`);
        this.currentConfiguration = response.data;
      } catch (error) {
        console.error("Failed to fetch configuration:", error);
        this.error = "Failed to fetch configuration";
      }
    },

    async fetchStereoImage() {
      try {
        const response = await axios.get("/stereo/image");
        if (response.data && response.data.image) {
          this.stereoImage = response.data.image;
        }
      } catch (error) {
        // Silently fail if no image available
      }
    },

    async fetchConfigTemplate(): Promise<StereoConfiguration | null> {
      try {
        const response = await axios.get("/stereo/template");
        return response.data;
      } catch (error) {
        console.error("Failed to fetch config template:", error);
        this.error = "Failed to fetch configuration template";
        return null;
      }
    },

    selectPair(pairId: string) {
      if (this.activePairs.includes(pairId)) {
        this.currentPairId = pairId;
        this.fetchConfiguration(pairId);
      }
    },

    setAutoRefresh(enabled: boolean) {
      this.autoRefresh = enabled;
    },

    clearError() {
      this.error = null;
    },

    // Create a default configuration with available cameras
    createDefaultConfig(leftCamera: string, rightCamera: string): StereoConfiguration {
      return {
        leftCameraName: leftCamera,
        rightCameraName: rightCamera,
        leftPosition: { x: -0.15, y: 0, z: 0, rotX: 0, rotY: 0, rotZ: 0 },
        rightPosition: { x: 0.15, y: 0, z: 0, rotX: 0, rotY: 0, rotZ: 0 },
        leftFOV: { horizontalFOV: 70, verticalFOV: 50 },
        rightFOV: { horizontalFOV: 70, verticalFOV: 50 }
      };
    }
  }
});
